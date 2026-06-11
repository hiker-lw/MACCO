import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_contrast_loss(image_features,
                                 text_features,
                                 logit_scale,
                                 device):
    num_logits = image_features.shape[0] # batch size
    # normal contrastive loss
    
    logits_per_img_to_text = logit_scale * image_features @ text_features.T
    logits_per_text_to_img = logit_scale * text_features @ image_features.T
    
    labels = torch.arange(num_logits, device=device, dtype=torch.long)
    normal_loss = (F.cross_entropy(logits_per_img_to_text, labels) + \
                F.cross_entropy(logits_per_text_to_img, labels))/2
    
    return normal_loss


def compute_contrast_loss_with_negative_text_image(image_features, text_features, logit_scale, device):
    num_logits = int(text_features.shape[0] // 2) # batch size
    # normal contrastive loss with negative text and image, compute the contrastive loss of negative image to text, 
    # and the contrastive loss of negative text to image
    
    logits_per_text_to_img = logit_scale * text_features @ image_features.T
    logits_per_img_to_text = logit_scale * image_features @ text_features.T
    
    labels_text_to_img = torch.arange(num_logits, device=device, dtype=torch.long).repeat(2)
    labels_img_to_text = torch.arange(num_logits, device=device, dtype=torch.long).repeat(2)
    normal_loss = (F.cross_entropy(logits_per_img_to_text, labels_img_to_text) + \
                F.cross_entropy(logits_per_text_to_img, labels_text_to_img))/2
    
    return normal_loss


def patchify(imgs: torch.Tensor, patch_size: int):
        """
        Split images into patches, following the official MAE implementation.
        imgs: [B, 3, H, W]
        returns: [B, num_patches, patch_size^2 * 3]
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x


class macco_clip_loss_text_image(nn.Module):
    # predict raw text signals, compute masked text reconstruction loss per sample, then average across batch.
    # predict raw pixel signals, compute masked image reconstruction loss per sample, then average across batch.
    # treat cls token of masked text and masked image as negative samples, 
    # also compute the contrastive loss of negative text to image and the contrastive loss of negative image to text
    # add the contrastive loss bewteen masked text and text, masked image and image (intra-modal contrastive loss)
    def __init__(self, args, norm_pix_loss: bool = True):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.mse_loss = nn.MSELoss(reduction='none')
        self.args = args
        if self.args.model == "ViT-B-32":
            self.patch_size = 32
        elif self.args.model == "ViT-B-16":
            self.patch_size = 16
        elif self.args.model == "ViT-L-14":
            self.patch_size = 14
    
    def get_loss_keys(self):
        return ["contrast_loss", "intra_modal_contrast_loss", "mask_loss_text", "mask_loss_image", "total_loss"]
    
    def _compute_contrast_loss(self, image_features, text_features, masked_image_features, masked_text_features, logit_scale):
        # normal contrastive loss
        device = image_features.device
        text_features = torch.cat([text_features, masked_text_features], dim=0)
        image_features = torch.cat([image_features, masked_image_features], dim=0)
        contrast_loss = compute_contrast_loss_with_negative_text_image(image_features, text_features, logit_scale, device)
        return contrast_loss
    
    def _compute_intra_modal_contrast_loss(self, image_features, text_features, masked_image_features, masked_text_features, logit_scale):
        # intra-modal contrastive loss
        device = image_features.device
        text_intra_modal_contrast_loss = compute_contrast_loss(masked_text_features, text_features, logit_scale, device)
        image_intra_modal_contrast_loss = compute_contrast_loss(masked_image_features, image_features, logit_scale, device)
        intra_modal_contrast_loss = 1/2 * (text_intra_modal_contrast_loss + image_intra_modal_contrast_loss)
        return intra_modal_contrast_loss
    
    def _compute_mask_loss_image(self, pred_image, target_image, image_mask_index, patch_size=32):
        image_mask_index = image_mask_index[:, 1:].to(pred_image.device)
        target_patches = patchify(target_image, patch_size)
        
        if self.norm_pix_loss:
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1e-6).sqrt()

        pred_image = pred_image.reshape(target_patches.shape[0], target_patches.shape[1], target_patches.shape[2])
        mask_loss_image = self.mse_loss(pred_image, target_patches)
        mask_loss_image = mask_loss_image.mean(dim=-1)  # [B, num_patches]

        # Per-sample loss computation.
        sample_loss_sum = (mask_loss_image * image_mask_index).sum(dim=1)        # [B]
        sample_mask_count = image_mask_index.sum(dim=1)               # [B]
        valid_samples = sample_mask_count > 0             # [B]
        
        # Compute the average loss for each valid sample.
        sample_loss = sample_loss_sum / sample_mask_count.clamp(min=1e-8)  # Avoid division by zero.
        
        # Exclude invalid samples from the final average.
        mask_loss_image = (sample_loss[valid_samples].sum() / 
                    (valid_samples.sum() + 1e-8))  # +1e-8 avoids division by zero when all samples are invalid.

        return mask_loss_image
    
    def _compute_mask_loss_text(self, pred_text, target_text, text_mask_index):
        device = target_text.device
        loss_fn = nn.NLLLoss(reduction='none')  # Keep per-token losses instead of reducing immediately.

        # Compute per-token NLL loss.
        token_loss = loss_fn(pred_text.permute(0, 2, 1), target_text)  # Match pred_text shape to target_text.
        # Shape: [batch_size, 77].

        # Compute loss only for masked tokens.
        masked_loss = token_loss * text_mask_index.to(device)  # [batch_size, 77]

        # Compute per-sample average reconstruction loss over masked tokens.
        per_sample_loss = masked_loss.sum(dim=1) / text_mask_index.sum(dim=1).clamp(min=1).to(device)  # [batch_size]

        # Average only over samples that have at least one masked token.
        valid_samples = text_mask_index.sum(dim=1) > 0
        if valid_samples.any():
            return per_sample_loss[valid_samples].mean()
        else:
            return torch.tensor(0.0, device=device)
    
    def forward(self, model_output_dict):
        image_features = model_output_dict["img_feat_cls_token"]
        text_features = model_output_dict["text_feat_cls_token"]
        masked_image_features = model_output_dict["masked_image_feat_cls_token"]
        masked_text_features = model_output_dict["masked_text_feat_cls_token"]

        pred_image = model_output_dict["pred_image"]
        target_image = model_output_dict["target_image"]

        pred_text = model_output_dict["pred_text"]
        target_text = model_output_dict["target_text"]

        logit_scale = model_output_dict["logit_scale"]
        image_mask_index = model_output_dict["image_mask_index"]
        text_mask_index = model_output_dict["text_mask_index"]

        contrast_loss = self._compute_contrast_loss(image_features, text_features, masked_image_features, masked_text_features, logit_scale)
        intra_modal_contrast_loss = self._compute_intra_modal_contrast_loss(image_features, text_features, masked_image_features, masked_text_features, logit_scale)
        mask_loss_image = self._compute_mask_loss_image(pred_image, target_image, image_mask_index, self.patch_size)
        mask_loss_text = self._compute_mask_loss_text(pred_text, target_text, text_mask_index)

        total_loss = self.args.contrast_loss_w * contrast_loss + self.args.intra_modal_contrast_loss_w * intra_modal_contrast_loss + \
            self.args.mask_image_loss_w * mask_loss_image + self.args.mask_text_loss_w * mask_loss_text
        
        loss = {
            "contrast_loss": contrast_loss,
            "intra_modal_contrast_loss": intra_modal_contrast_loss,
            "mask_loss_text": mask_loss_text,
            "mask_loss_image": mask_loss_image,
            "total_loss": total_loss
        }
        
        return loss
