from itertools import repeat
import collections.abc
import os
import ast
import torch
import random
import math
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
from .tokenizer import tokenize, _tokenizer



def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)

# NOTE: below are some customized functions
# some mask functions
def heavy_masking_img(x, mask_size):
        """
        Perform random square masking by per-sample shuffling.
        mask_size: (h_mask, w_mask), mask window size, tuple
        """
        N, L, D = x.shape  # batch, length, dim
        h_mask, w_mask = mask_size # square mask size
        h_patch = int(math.sqrt(L))
        index_keep = torch.zeros([N, L-h_mask*w_mask], device=x.device)
        for index in range(N):
                # mask start index list
                mask_start_list = [i*h_patch+j for i in range(0,h_patch-h_mask+1) for j in range(0,h_patch-w_mask+1)]
                # mask start index
                random_mask_start = random.randint(0,len(mask_start_list))
                # mask index list for a single image
                mask_index_single = [random_mask_start+i*h_patch+j for i in range(0,h_mask) for j in range(0,w_mask)]
                # not masked index for a single image
                index_all_single = [k for k in range(L)]
                index_keep_single = [item for item in index_all_single if item not in mask_index_single]
                index_keep[index] = index_keep_single

        x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked

def block_masking_img(x, mask_ratio):
        """
        block masking (block : 2 * 2 patches)
        """
        N, L, D = x.shape  # batch, length, dim
        sample_index = [0, 2, 4, 6, 14, 16, 18, 20, 28, 30, 32, 34, 42, 44, 46, 48]
        mask = torch.zeros((N, L), device=x.device)
        for i in range(N):
            mask_block_index = random.sample(sample_index, int(len(sample_index) * mask_ratio))
            for index in mask_block_index:
                  if index in [6, 20, 34]:
                        mask[i, [index, index+7]] = 1
                  elif index in [42, 44, 46]:
                        mask[i, [index, index+1]] = 1
                  elif index == 48:
                        mask[i, index] = 1
                  else:
                        mask[i, [index, index+1, index+7, index+8]] = 1

        return mask

def random_masking_img(x, mask_ratio):
        # code borrowed from MAE (https://github.com/facebookresearch/mae)
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

def random_masking_text(tokens_list, mask_ratio):
        # code modified from BERT (https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch)
        """
        perform random masking to text tokens
        tokens_list: [N, L], N is the text batchsize, L is the token number of a text
        """

        """
        mask_index = torch.zeros(tokens_list.shape, dtype=bool, device=tokens_list.device)
        for index, tokens_single_text in enumerate(tokens_list):
             # enumerate real tokens of the text excluding the padding tokens
             for i, _ in enumerate(tokens_single_text):
                if i >=1 and i < tokens_single_text.argmax():
                    # mask each token by a certain probability
                    prob = random.random()
                    if prob < mask_ratio:
                        mask_index[index][i] = True
        """

        mask_index = torch.zeros(tokens_list.shape, dtype=bool, device=tokens_list.device)
        for index, tokens_single_text in enumerate(tokens_list):
            valid_token_num = tokens_single_text.argmax() - 1
            mask_token_num = int(mask_ratio * valid_token_num)
            mask_index_single_text = random.sample([x for x in range(1, valid_token_num+1)], mask_token_num)
            for i in mask_index_single_text:
                mask_index[index][i] = True

        return mask_index

def masking_text_padding(tokens_list):    
    mask_index = torch.zeros(tokens_list.shape, dtype=bool, device=tokens_list.device)
    for index, tokens_single_text in enumerate(tokens_list):
        valid_token_num = tokens_single_text.argmax() + 1
        mask_index[index][valid_token_num: ] = True
    return mask_index

def manual_mask_text(tokens_list, text_masked_index):
        """
        perform manual masking to text tokens

        """
        mask_index = torch.zeros(tokens_list.shape, dtype=torch.bool, device=tokens_list.device)
        for index in text_masked_index:
            mask_index[0][index] = True
        
        return mask_index

def patchify(imgs, patch_size):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def visualize_masked_text_single(mask_index, text_tokens):
    token_without_padding = text_tokens[0][1:text_tokens[0].argmax(dim=-1)].tolist()
    for index in range(1, len(token_without_padding) + 1):
        if mask_index[0][index] == True:
            token_without_padding[index-1] = 49396
            
    text = _tokenizer.decode(token_without_padding)
    text_masked = text.replace("foxtv", "<mask>")
    
    return text_masked

def visualize_text_single(text_tokens):
    token_without_padding = text_tokens[0][1:text_tokens[0].argmax(dim=-1)].tolist()
            
    text = _tokenizer.decode(token_without_padding)
    
    return text

def decode_singe_token(token):
            
    word = _tokenizer.decode([token])
    
    return word


def visualize_input_image_single(x):
        # recover raw image
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        ori_img = x[0].permute(1, 2, 0).data.cpu().numpy()
        ori_img = ori_img * std + mean
        ori_img = np.clip(ori_img, 0, 1)
        ori_img = np.uint8(255 * ori_img)
        return ori_img

def chose_mask_word(mask_num, word_list, texts):
        # only mask relation word
        if len(word_list) == 0:
            mask_index = []
        else:
            if mask_num == "not_fixed":
                chosen_word = word_list
            else:
                if mask_num > len(word_list):
                    chosen_word = word_list
                else:
                    chosen_word = random.sample(word_list, mask_num)
            mask_index = []
            for word in chosen_word:
                word_token_num = torch.count_nonzero(tokenize([str(word)])[0]) - 2
                if word_token_num == 1:
                    for i, value in enumerate(texts.tolist()):
                        if value == tokenize([str(word)])[0][1]:
                            mask_index.append(i)
                elif word_token_num > 1:
                    for i, value in enumerate(texts.tolist()):
                        if value == tokenize([str(word)])[0][1]:
                            mask_index = mask_index + [i+k for k in range(0, word_token_num)]
        return mask_index


def compute_test_loss(model, data, device, loss_factory, args):
    # compute test loss
    with torch.no_grad():
        dataloader_val = data['val'].dataloader
        normal_loss_val = 0
        mask_loss_text_val = 0
        total_loss_val = 0
        for _, batch in enumerate(dataloader_val):
            images, texts, text_mask_index = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            output_dict = model(images, texts, text_mask_index)
            loss_val = loss_factory[args.loss_fn](output_dict, args)()
            normal_loss_val = normal_loss_val + loss_val['normal_loss'] * len(images)
            mask_loss_text_val = mask_loss_text_val + loss_val['mask_loss_text'] * len(images)
            total_loss_val = total_loss_val + loss_val['total_loss'] * len(images)
        
        normal_loss_val = normal_loss_val / len(dataloader_val.dataset)
        mask_loss_text_val = mask_loss_text_val / len(dataloader_val.dataset)
        total_loss_val = total_loss_val / len(dataloader_val.dataset)

        val_loss_data = {}
        val_loss_data['normal_loss_val'] = normal_loss_val
        val_loss_data['mask_loss_text_val'] = mask_loss_text_val
        val_loss_data['total_loss_val'] = total_loss_val

    return val_loss_data

def vis_pred_during_training(model, args, preprocess_train):
    # visualize prediction words during training
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    df = pd.read_csv(os.path.join(root_path, args.vis_data), sep="\t", \
                     converters={"noun_list":ast.literal_eval, 'relation_word_list':ast.literal_eval, 'adjective_word_list':ast.literal_eval})
    image_path = df['filepath'].tolist()
    captions = df['title'].tolist()
    noun_list = df['noun_list'].tolist()
    relation_word_list = df['relation_word_list'].tolist()
    adjective_word_list = df['adjective_word_list'].tolist()
    transforms = preprocess_train
    noun_mask_num = args.noun_mask_num
    relation_mask_num = args.relation_mask_num
    adjective_mask_num = args.adjective_mask_num
    mask_word_type = args.mask_word_type

    vis_data = []
    for idx in range(len(image_path)):
        images = transforms(Image.open(os.path.join(root_path, "datasets", str(image_path[idx]))))
        caption = captions[idx]
        caption = caption.strip()
        texts = tokenize([str(caption)])[0]
        if mask_word_type == "noun":
            # only mask noun word
            mask_index = chose_mask_word(mask_num=noun_mask_num, word_list=noun_list[idx], texts=texts)
            
        elif mask_word_type == "relation":
            # only mask relation word
            mask_index = chose_mask_word(mask_num=relation_mask_num, word_list=relation_word_list[idx], texts=texts)
        
        elif mask_word_type == "adjective":
            # only mask adjective word
            mask_index = chose_mask_word(mask_num=adjective_mask_num, word_list=adjective_word_list[idx], texts=texts)

        elif mask_word_type == "noun_and_relation":
            # mask noun and relation word
            mask_index_noun = chose_mask_word(mask_num=noun_mask_num, word_list=noun_list[idx], texts=texts)
            mask_index_relation = chose_mask_word(mask_num=relation_mask_num, word_list=relation_word_list[idx], texts=texts)
            mask_index = mask_index_noun + mask_index_relation
        
        elif mask_word_type == "noun_and_adjective":
            # mask noun and adjective
            mask_index_noun = chose_mask_word(mask_num=noun_mask_num, word_list=noun_list[idx], texts=texts)
            mask_index_adjective = chose_mask_word(mask_num=adjective_mask_num, word_list=adjective_word_list[idx], texts=texts)
            mask_index = mask_index_noun + mask_index_adjective
        
        elif mask_word_type == "relation_and_adjective":
            # mask relation and adjective
            mask_index_relation = chose_mask_word(mask_num=relation_mask_num, word_list=relation_word_list[idx], texts=texts)
            mask_index_adjective = chose_mask_word(mask_num=adjective_mask_num, word_list=adjective_word_list[idx], texts=texts)
            mask_index = mask_index_relation + mask_index_adjective
        
        elif mask_word_type == "noun_and_relation_and_adjective":
            # mask noun, relation and adjective
            mask_index_noun = chose_mask_word(mask_num=noun_mask_num, word_list=noun_list[idx], texts=texts)
            mask_index_relation = chose_mask_word(mask_num=relation_mask_num, word_list=relation_word_list[idx], texts=texts)
            mask_index_adjective = chose_mask_word(mask_num=adjective_mask_num, word_list=adjective_word_list[idx], texts=texts)
            mask_index = mask_index_noun + mask_index_relation + mask_index_adjective
        
        mask_status = torch.zeros(len(texts)).bool()
        mask_status[mask_index] = True
        vis_data.append((images.unsqueeze(0), texts.unsqueeze(0), mask_status.unsqueeze(0)))

    device = torch.device(args.device)

    model.eval()
    input_raw_text_list = []
    mask_raw_text_list = []
    pred_raw_text_list = []
    confidence_score_list = []
    for _, sample in enumerate(vis_data):
        confidence_score = []
        images, texts, text_mask_index = sample
        texts_ori = torch.empty_like(texts).copy_(texts)
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        output_dict = model(images, texts, text_mask_index)
        pred_text = output_dict['pred_text']
        pred_text_score = torch.nn.Softmax(dim=-1)(pred_text)
        pred_text = pred_text_score.argmax(dim=-1)
        mask_index_list = []
        for i in range(len(text_mask_index[0])):
            if text_mask_index[0][i] == True:
                mask_index_list.append(i)
        mask_index_list = torch.tensor(mask_index_list).unsqueeze(0)
        for i in range(len(texts)):
            for mask_index in mask_index_list[0]:
                texts_ori[i][mask_index] = pred_text[i][mask_index]
                confidence_score.append(round(pred_text_score[i][mask_index][pred_text[i][mask_index]].item(), 2))

        input_raw_text = visualize_text_single(texts)
        mask_raw_text = visualize_masked_text_single(text_mask_index, texts)
        pred_text_token = texts_ori
        pred_raw_text = visualize_text_single(pred_text_token)
        input_raw_text_list.append(input_raw_text)
        mask_raw_text_list.append(mask_raw_text)
        pred_raw_text_list.append(pred_raw_text)
        confidence_score_list.append(confidence_score)
    
    save_path = os.path.join(root_path, args.logs, args.name, 'vis_pred_during_training')
    os.makedirs(save_path, exist_ok=True)
    for i in range(len(vis_data)):
        # write the prediction into txt file
        with open(os.path.join(save_path, f"{i}.txt"), 'a+') as file:
            file.write(f"input_raw_text: {input_raw_text_list[i]}, mask_raw_text: {mask_raw_text_list[i]}, pred_raw_text: {pred_raw_text_list[i]}, confidence_score: {confidence_score_list[i]}\n")
        file.close()