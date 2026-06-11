import torch
from torchvision.transforms import functional as F
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

class RandomResizedCrop_with_tracking(RandomResizedCrop):
    """RandomResizedCrop variant that records the sampled transform parameters."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_params = None  # Store the latest transform parameters.

    def forward(self, img):
        # Sample crop parameters.
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        self.last_params = (i, j, h, w, img.height, img.width)
        
        return F.resized_crop(
            img, i, j, h, w, self.size, self.interpolation
        )

class BBoxAdapter:
    def __init__(self, image_size=224):
        self.image_size = image_size
        
    def transform_bbox(self, orig_bbox, transform_params):
        """
        Transform a bounding box according to the sampled augmentation parameters.
        :param orig_bbox: [x1,y1,x2,y2] coordinates in the original image
        :param transform_params: (i,j,h,w, orig_h, orig_w)
        :return: transformed coordinates [x1,y1,x2,y2]
        """
        i, j, h, w, orig_h, orig_w = transform_params
        # Compute scaling factors.
        scale_h = self.image_size / h
        scale_w = self.image_size / w
        
        # Convert coordinates from the original image to the crop, then resize.
        x1 = max(orig_bbox[0] - j, 0) * scale_w
        y1 = max(orig_bbox[1] - i, 0) * scale_h
        x2 = min(orig_bbox[2] - j, w) * scale_w
        y2 = min(orig_bbox[3] - i, h) * scale_h
        
        return [
            x1, y1,
            x2, y2
        ]

def _convert_to_rgb(image):
    return image.convert('RGB')
def create_transform_with_tracking(image_size=224):
    crop = RandomResizedCrop_with_tracking(
        image_size, 
        scale=(0.9, 1.0), 
        interpolation=InterpolationMode.BICUBIC
    )
    mean = (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
    std = (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
    def get_transform_params():
        return crop.last_params
    
    transforms = Compose([
        crop,
        _convert_to_rgb,
        ToTensor(),
        Normalize(mean, std)
    ])
    
    return transforms, get_transform_params

def bbox_to_patch_mask(transformed_bboxes, image_size=224, patch_size=32):
    """
    Map transformed bounding boxes to a ViT patch mask.
    :param transformed_bboxes: list of transformed [x1,y1,x2,y2] coordinates
    :param image_size: input image size
    :param patch_size: patch size
    :return: mask_flag (Tensor [n_tokens]) 
    """
    # Compute grid parameters.
    grid_size = image_size // patch_size
    n_patches = grid_size ** 2
    
    # Initialize mask, including the class-token position.
    mask_flag = torch.zeros(n_patches + 1, dtype=torch.bool)
    
    for bbox in transformed_bboxes:
        x1, y1, x2, y2 = bbox
        
        # Compute the covered patch range.
        col_start = int(x1 // patch_size)
        col_end = int((x2 - 1e-5) // patch_size)  # Avoid floating-point boundary spillover.
        row_start = int(y1 // patch_size)
        row_end = int((y2 - 1e-5) // patch_size)
        
        # Clamp to valid patch coordinates.
        col_start = max(0, min(col_start, grid_size-1))
        col_end = max(0, min(col_end, grid_size-1))
        row_start = max(0, min(row_start, grid_size-1))
        row_end = max(0, min(row_end, grid_size-1))
        
        # Fill the mask region.
        for row in range(row_start, row_end+1):
            for col in range(col_start, col_end+1):
                # ViT patch tokens are flattened in row-major order.
                patch_idx = row * grid_size + col
                mask_flag[patch_idx + 1] = True  # +1 skips the class token.

    return mask_flag
