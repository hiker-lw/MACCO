import math
import random
from collections import OrderedDict
from dataclasses import dataclass
import logging
import sys
from typing import Tuple, Union, Callable, Optional
from torch.utils.checkpoint import checkpoint
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from open_clip_code.model import LayerNorm, ResidualAttentionBlock

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_x = LayerNorm(d_model)
        self.ln_y = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_post = LayerNorm(d_model)

        self.attn_weight = None

    def cross_attention(self, x: torch.Tensor, y: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        CA_out = self.attn(x, y, y, need_weights=True, attn_mask=attn_mask)
        self.attn_weight = CA_out[1]
        return CA_out[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.cross_attention(self.ln_x(x), self.ln_y(y), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_post(x))
        return x


class Image_Predictor(nn.Module):
    def __init__(self, 
                 width: int, 
                 self_attn_layers: int,
                 cross_attn_layers: int,
                 heads: int,
                 patch_size: int = 32,
                 in_chans: int = 3
                 ):
        super().__init__()

        self.self_attn_layers = self_attn_layers
        self.cross_attn_layers = cross_attn_layers
        self.width = width

        self.num_patches = (224 // patch_size) ** 2
        self.decoder_embed = nn.Linear(width, width, bias=True)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, width), requires_grad=False)

        self.self_attn_blocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads)
            for _ in range(self.self_attn_layers)
        ])
        self.cross_attn_blocks = nn.ModuleList([
            ResidualCrossAttentionBlock(width, heads)
            for _ in range(self.cross_attn_layers)
        ])
        self.ln = nn.LayerNorm(width)
        
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, patch_size**2 * in_chans)  # Each token predicts pixels for one patch.
        )
        self._init_weights()
    
    def _init_weights(self):
        # Use the same initialization strategy as MAE.
        nn.init.xavier_uniform_(self.proj[-1].weight, gain=1/self.proj[-1].in_features**0.5)
        nn.init.zeros_(self.proj[-1].bias)

        pos_embedding = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], int(self.num_patches**.5), cls_token=False)
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embedding).float().unsqueeze(0))

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_cls: torch.Tensor, texts: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        '''
        Input:
            x: masked image features (all tokens) from CLIP image encoder, [b, 49, 512]
            y: text features (all tokens) from CLIP text encoder, [b, 77, 512]
            texts: text tensor input to CLIP text encoder, [b, 77]
            attn_mask: full attention or causal attention mask between the tokens, default is None (full attention)
        Output:
            x: predicted pixel value, [b, 49, patch**2*3]
        '''
        y_cls = y[torch.arange(texts.shape[0]), texts.argmax(dim=-1)]
        y = (y + y_cls[:, None, :]) / 2

        x = (x + x_cls[:, None, :]) / 2

        x = self.decoder_embed(x)
        x = x + self.pos_embedding

        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        
        for self_attn_block in self.self_attn_blocks:
            x = self_attn_block(x)
        
        for cross_attn_block in self.cross_attn_blocks:
            x = cross_attn_block(x, y, attn_mask)
            
        x = x.permute(1, 0, 2)
        x = self.ln(x)

        x = self.proj(x)  # [B, num_patches, patch_size^2 * 3]

        return x

class Text_Predictor(nn.Module):
    def __init__(self, 
                 width: int, 
                 self_attn_layers: int,
                 cross_attn_layers: int,
                 heads: int
                 ):
        super().__init__()

        self.self_attn_layers = self_attn_layers
        self.cross_attn_layers = cross_attn_layers
        self.width = width
        self.self_attn_blocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads)
            for _ in range(self.self_attn_layers)
        ])
        self.cross_attn_blocks = nn.ModuleList([
            ResidualCrossAttentionBlock(width, heads)
            for _ in range(self.cross_attn_layers)
        ])
        self.ln = nn.LayerNorm(width)
        # vocabulary size: 49408
        self.linear = nn.Linear(width, 49408, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)
        torch.nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, y_cls: torch.Tensor, texts: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        '''
        Input:
            x: masked text features (all tokens) from CLIP text encoder, [b, 77, 512]
            y: image features (all tokens) from CLIP image encoder, [b, 49, 512]
            texts: text tensor input to CLIP text encoder, [b, 77]
            attn_mask: full attention or causal attention mask between the tokens, default is None (full attention)
        Output:
            x: prediction probability of every word in the vocabulary, [b, 77, 49408]
        '''
        x_cls = x[torch.arange(texts.shape[0]), texts.argmax(dim=-1)]
        x = (x + x_cls[:, None, :]) / 2
        y = (y + y_cls[:, None, :]) / 2
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)

        
        for self_attn_block in self.self_attn_blocks:
            x = self_attn_block(x)

        for cross_attn_block in self.cross_attn_blocks:
            x = cross_attn_block(x, y, attn_mask)
        
        x = x.permute(1, 0, 2)
        x = self.ln(x)
        x = self.linear(x) # classification 
        x = self.softmax(x)
        return x

class MACCO_CLIP_text_image(nn.Module):
    '''
    Cross-Modal Masked Composition Concept Modeling
    '''
    def __init__(self, CLIP, args, width: int=512, heads: int=8, output_dim: int=512):
        super(MACCO_CLIP_text_image, self).__init__()
        # pretrained CLIP model
        self.CLIP = CLIP
        self.args = args
        self.logit_scale = self.CLIP.logit_scale

        # replace encode_text, encode_image and forward function
        self.CLIP.encode_text = self.clip_encode_text.__get__(self.CLIP)
        self.CLIP.encode_image = self.clip_encode_image.__get__(self.CLIP)
        self.CLIP.visual.forward = self.clip_vit_forward.__get__(self.CLIP.visual)
        self.CLIP.forward = self.clip_forward.__get__(self.CLIP)

        if self.args.model == "ViT-B-32":
            self.patch_size = 32
            width = 512
        elif self.args.model == "ViT-B-16":
            self.patch_size = 16
            width = 512
        elif self.args.model == "ViT-L-14":
            self.patch_size = 14
            width = 768

        # prediction layer for text
        self.prediction_text = Text_Predictor(width, args.text_predictor_self_attn_layers, args.text_predictor_cross_attn_layers, heads)
        # prediction layer for image
        self.prediction_image = Image_Predictor(width, args.image_predictor_self_attn_layers, args.image_predictor_cross_attn_layers, heads, patch_size=self.patch_size)

        # text mask token and image mask token, randomly initialized
        text_token_dim = CLIP.token_embedding.weight.shape[1]
        self.text_mask_token = nn.Parameter(torch.randn(text_token_dim))
        img_token_dim = CLIP.visual.positional_embedding.shape[1]
        self.img_mask_token = nn.Parameter(torch.randn(img_token_dim))
        self.init_parameters()

    def init_parameters(self):
        # code borrowed from neg_clip
        # init the mask token
        nn.init.normal_(self.text_mask_token, std=0.02)
        nn.init.normal_(self.img_mask_token, std=0.02)

        # Initialize parameters for prediction_text block
        if self.prediction_text.self_attn_layers > 0:
            self_attn_proj_std = (self.prediction_text.width ** -0.5) * ((2 * self.prediction_text.self_attn_layers) ** -0.5)
        if self.prediction_text.cross_attn_layers > 0:
            cross_attn_proj_std = (self.prediction_text.width ** -0.5) * ((2 * self.prediction_text.cross_attn_layers) ** -0.5)
        attn_std = self.prediction_text.width ** -0.5
        fc_std = (2 * self.prediction_text.width) ** -0.5
        for cross_attn_block in self.prediction_text.cross_attn_blocks:
            nn.init.normal_(cross_attn_block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(cross_attn_block.attn.out_proj.weight, std=cross_attn_proj_std)
            nn.init.normal_(cross_attn_block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(cross_attn_block.mlp.c_proj.weight, std=cross_attn_proj_std)
        
        for self_attn_block in self.prediction_text.self_attn_blocks:
            nn.init.normal_(self_attn_block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self_attn_block.attn.out_proj.weight, std=self_attn_proj_std)
            nn.init.normal_(self_attn_block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(self_attn_block.mlp.c_proj.weight, std=self_attn_proj_std)
        
        # Initialize parameters for prediction_image block
        if self.prediction_image.self_attn_layers > 0:
            self_attn_proj_std = (self.prediction_image.width ** -0.5) * ((2 * self.prediction_image.self_attn_layers) ** -0.5)
        if self.prediction_image.cross_attn_layers > 0:
            cross_attn_proj_std = (self.prediction_image.width ** -0.5) * ((2 * self.prediction_image.cross_attn_layers) ** -0.5)
        attn_std = self.prediction_image.width ** -0.5
        fc_std = (2 * self.prediction_image.width) ** -0.5
        for cross_attn_block in self.prediction_image.cross_attn_blocks:
            nn.init.normal_(cross_attn_block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(cross_attn_block.attn.out_proj.weight, std=cross_attn_proj_std)
            nn.init.normal_(cross_attn_block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(cross_attn_block.mlp.c_proj.weight, std=cross_attn_proj_std)
        
        for self_attn_block in self.prediction_image.self_attn_blocks:
            nn.init.normal_(self_attn_block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self_attn_block.attn.out_proj.weight, std=self_attn_proj_std)
            nn.init.normal_(self_attn_block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(self_attn_block.mlp.c_proj.weight, std=self_attn_proj_std)
    
    def clip_encode_text(self, text, text_mask_index, text_mask_token):
        # makesure text_mask_index is bool type
        text_mask_index = text_mask_index.to(dtype=torch.bool)

        # initialize the key mask matrix (dim: [2N,L])
        key_mask = torch.zeros([text.shape[0] * 2, text.shape[1]], dtype=torch.bool, device=text.device)
        
        # fill the lower half of the key mask matrix with the generated mask index
        key_mask[text.shape[0]:] = text_mask_index

        x_full_mask = torch.cat([text, text], dim=0) # 2N * L * D

        x_full_mask = self.token_embedding(x_full_mask)  # [batch_size, n_ctx, d_model]
        
        # fill the masked position with mask_token
        # x_full_and_masked[key_mask] = mask_token[None, :]
        x_full_mask = x_full_mask * (~key_mask[:, :, None]) + \
                text_mask_token[None, None, :] * key_mask[:, :, None]

        x_full_mask = x_full_mask + self.positional_embedding
        x_full_mask = x_full_mask.permute(1, 0, 2)  # NLD -> LND
        x_full_mask = self.transformer(x_full_mask, attn_mask=self.attn_mask)
        x_full_mask = x_full_mask.permute(1, 0, 2)  # LND -> NLD
        x_full_mask = self.ln_final(x_full_mask)

        # projection
        # x_full_and_masked.shape = [batch_size * 2 , n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eot_token_idx = text.argmax(dim=-1)
        x_full_mask_proj = x_full_mask @ self.text_projection

        # all text token feature
        x_all_token = x_full_mask_proj[0:text.shape[0]]
        x_masked_all_token = x_full_mask_proj[text.shape[0]: text.shape[0]*2]

        # take CLS token from the eot position
        x = x_all_token[torch.arange(text.shape[0]), eot_token_idx]
        x_masked = x_masked_all_token[torch.arange(text.shape[0]), eot_token_idx]

        return x, x_all_token, x_masked, x_masked_all_token

    def clip_encode_image(self, image, image_mask_index, img_mask_token):
        return self.visual(image, image_mask_index, img_mask_token)

    def clip_vit_forward(self, x, image_mask_index, img_mask_token):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        # initialize the key mask matrix (dim: [2N, L])
        key_mask = torch.zeros([x.shape[0] * 2, x.shape[1]], dtype=torch.bool, device=x.device)
        
        # fill the lower half of the key mask matrix with the generated mask index
        key_mask[x.shape[0]:] = image_mask_index

        x_full_mask = torch.cat([x, x], dim=0) # 2N * L * D

        # fill the masked position with mask_token
        x_full_mask = x_full_mask * (~key_mask[:, :, None]) + \
                img_mask_token[None, None, :] * key_mask[:, :, None]
        
        x_full_mask = x_full_mask + self.positional_embedding.to(x_full_mask.dtype)
        x_full_mask = self.ln_pre(x_full_mask)

        x_full_mask = x_full_mask.permute(1, 0, 2)  # NLD -> LND
        x_full_mask = self.transformer(x_full_mask)
        x_full_mask = x_full_mask.permute(1, 0, 2)  # LND -> NLD

        x_full_mask = self.ln_post(x_full_mask) # modified here to output both cls token features and image patch features

        if self.proj is not None:
            x_full_mask_proj = x_full_mask @ self.proj
            cls_token = x_full_mask_proj[:x.shape[0], 0]
            cls_token_masked = x_full_mask_proj[x.shape[0]:, 0]
            patch_token = x_full_mask_proj[:x.shape[0], 1:]
            patch_token_masked = x_full_mask_proj[x.shape[0]:, 1:]

        return cls_token, patch_token, cls_token_masked, patch_token_masked
    
    def clip_forward(self, image, text, image_mask_index=None, img_mask_token=None, text_mask_index=None, text_mask_token=None,):
        if image is None:
            self.encode_text(text, text_mask_index, text_mask_token)
        elif text is None:
            return self.encode_image(image, image_mask_index, img_mask_token)

        # get text and image feature
        text_feature, text_feature_all, masked_text_feature, masked_text_feature_all = self.encode_text(text, text_mask_index, text_mask_token)
        image_feature, image_feature_all, masked_image_feature, masked_image_feature_all = self.encode_image(image, image_mask_index, img_mask_token)

        image_feature_before_norm = image_feature
        masked_image_feature_before_norm = masked_image_feature

        text_feature = F.normalize(text_feature, dim=-1)
        image_feature = F.normalize(image_feature, dim=-1)
        masked_text_feature = F.normalize(masked_text_feature, dim=-1)
        masked_image_feature = F.normalize(masked_image_feature, dim=-1)


        result_dict = {
            "image_feature": image_feature,
            "text_feature": text_feature,
            "image_feature_before_norm": image_feature_before_norm,
            "masked_image_feature_before_norm": masked_image_feature_before_norm,
            "image_feature_all": image_feature_all,
            "text_feature_all": text_feature_all,
            "masked_text_feature": masked_text_feature,
            "masked_text_feature_all": masked_text_feature_all,
            "masked_image_feature": masked_image_feature,
            "masked_image_feature_all": masked_image_feature_all,
            "logit_scale": self.logit_scale.exp(),
        }

        return result_dict
    
    def forward(self, images, texts, image_mask_index, text_mask_index):
        '''
        Input:
            images: image tensor input to CLIP image encoder, [b, 3, 224, 224]
            texts: text tensor input to CLIP text encoder, [b, 77]
            text_mask_index: mask for text, every item is a boolean value (True: mask, False: not mask), [b, 77]
            image_mask_index: mask for image, every item is a boolean value (True: mask, False: not mask), [b, 50]
        Output:
            output: a dictionary
            output["img_feat_cls_token"]: image CLS token from CLIP image encoder, [b, 512]
            output["img_feat_all_token"]: image all tokens from CLIP image encoder, [b, 49, 512]
            output["text_feat_cls_token"]: text CLS token from CLIP text encoder, [b, 512]
            output["text_feat_all_token"]: text all tokens from CLIP text encoder, [b, 77, 512]
            output["masked_text_feat_cls_token"]: masked text CLS token from CLIP text encoder, [b, 512]
            output["masked_text_feat_all_token"]: masked text all tokens from CLIP text encoder, [b, 77, 512]
            output["pred_text"]: prediction probability of every word in the vocabulary, [b, 77, 49408]
            output["target_text"]: original input text, [b, 77]
            output["masked_image_feat_cls_token"]: masked image CLS token from CLIP image encoder, [b, 512]
            output["masked_image_feat_all_token"]: masked image all tokens from CLIP image encoder, [b, 49, 512]
            output["pred_image"]: predict patch pixel value, [b, 49, patch_size**2*3]
            output["target_image"]: original input image, [b, 3, 224, 224]
            output["images"]: input image tensor, [b, 3, 224, 224]
            output["texts"]: input text tensor, [b, 77]
            output["logit_scale"]: 1/temperature
            output["image_mask_index"]: mask for image, every item is a boolean value (True: mask, False: not mask), [b, 50]
        '''

        # output from CLIP
        CLIP_out = self.CLIP(images, texts, image_mask_index, self.img_mask_token, text_mask_index, self.text_mask_token)
        image_feature = CLIP_out["image_feature"]
        text_feature = CLIP_out["text_feature"]
        image_feature_before_norm = CLIP_out["image_feature_before_norm"]
        masked_image_feature_before_norm = CLIP_out["masked_image_feature_before_norm"]
        image_feature_all = CLIP_out["image_feature_all"]
        text_feature_all = CLIP_out["text_feature_all"]
        masked_text_feature = CLIP_out["masked_text_feature"]
        masked_text_feature_all = CLIP_out["masked_text_feature_all"]
        masked_image_feature = CLIP_out["masked_image_feature"]
        masked_image_feature_all = CLIP_out["masked_image_feature_all"]
        logit_scale = CLIP_out["logit_scale"]

        # predict the text
        pred_text = self.prediction_text(masked_text_feature_all, image_feature_all.detach(), image_feature_before_norm.detach(), texts)
        target_text = texts

        # predict the image
        pred_image = self.prediction_image(masked_image_feature_all.detach(), text_feature_all, masked_image_feature_before_norm.detach(), texts)
        target_image = images

        # normalization
        image_feature_all = F.normalize(image_feature_all, dim=-1)
        text_feature_all = F.normalize(text_feature_all, dim=-1)

        logit_scale = self.logit_scale.exp()

        output = {"img_feat_cls_token": image_feature,
                  "img_feat_all_token": image_feature_all,
                  "text_feat_cls_token": text_feature,
                  "text_feat_all_token": text_feature_all,

                  "masked_text_feat_cls_token": masked_text_feature,
                  "masked_text_feat_all_token": masked_text_feature_all,

                  "pred_text": pred_text,
                  "target_text": target_text,

                  "masked_image_feat_cls_token": masked_image_feature,
                  "masked_image_feat_all_token": masked_image_feature_all,

                  "pred_image": pred_image,
                  "target_image": target_image,

                  "images": images,
                  "texts":texts,

                  "logit_scale": logit_scale,
                  "image_mask_index": image_mask_index,
                  "text_mask_index": text_mask_index}

        return output


if __name__ == '__main__':
     pass
