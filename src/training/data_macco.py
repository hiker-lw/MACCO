import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value
import ast
import random
import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from training.utils import BBoxAdapter, bbox_to_patch_mask

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip_code import tokenize
from open_clip_code import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

random.seed(2024)

class CsvDataset_mask_text(Dataset):
    """
    Mask compositional concepts in text.
    """
    def __init__(self, args, input_filename, transforms, img_key, caption_key, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}.')
        self.args = args
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets")
        df = pd.read_csv(os.path.join(self.root_path, input_filename), sep=sep, \
                        converters={"object_list":ast.literal_eval, "relation_list":ast.literal_eval, \
                                    "attribute_list":ast.literal_eval, "neg_caption":ast.literal_eval, "neg_image":ast.literal_eval})
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.hard_captions = df["neg_caption"].tolist()
        self.noun_list = df['object_list'].tolist()
        self.relation_word_list = df['relation_list'].tolist()
        self.adjective_word_list = df['attribute_list'].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)
    
    def get_word_token_pos_in_text_token(self, word, text):
        # get the word token position in all text token
        text_token_list = text.clone().tolist()
        word_token = _tokenizer.encode(word)
        start_idx = -1
        for idx in range(len(text)):
            if text_token_list[idx: idx + len(word_token)] == word_token:
                start_idx = idx
                break
        if start_idx != -1:
            return [start_idx + i for i in range(len(word_token))]
        else:
            # raise Exception("Not found matched word in the text!!!")
            return []

    def get_token_flag(self, idx, texts):
        comp_concept_token_idx = [] # Indices of relation and attribute tokens in the full caption.
        object_token_idx = [] # Indices of object tokens in the full caption.
        attribute_word_list = self.adjective_word_list[idx]
        relation_word_list = self.relation_word_list[idx]
        object_word_list = self.noun_list[idx]
        
        for word in relation_word_list:
            # if word.lower() != "the" and word.lower() != "in" and word.lower() != "on":
            if word.lower() != "the":
                comp_concept_token_idx += self.get_word_token_pos_in_text_token(word, texts)
        
        for word in attribute_word_list:
            if word.lower() != "the":
                comp_concept_token_idx += self.get_word_token_pos_in_text_token(word, texts)

        for word in object_word_list:
            if word.lower() != "the":
                object_token_idx += self.get_word_token_pos_in_text_token(word, texts)
        
        comp_concept_token_flag = torch.zeros(len(texts)).bool()
        comp_concept_token_flag[comp_concept_token_idx] = True

        object_token_flag = torch.zeros(len(texts)).bool()
        object_token_flag[object_token_idx] = True

        return comp_concept_token_flag, object_token_flag

    def __getitem__(self, idx):

        images = self.transforms(Image.open(os.path.join(self.data_path, str(self.images[idx]))))
        caption = self.captions[idx]
        caption = caption.strip()
        texts = tokenize([str(caption)])[0]
        
        try:
            chosen_caption = random.choice(self.hard_captions[idx])
            hard_captions = tokenize([str(chosen_caption)])[0] # hard negative caption for texts
        except:
            hard_captions = tokenize([str("A photo of full black image.")])[0]

        
        comp_concept_token_flag, object_token_flag = self.get_token_flag(idx, texts)
        return images, texts, hard_captions, comp_concept_token_flag, object_token_flag

class CsvDataset_mask_image(Dataset):
    """
    Mask compositional concepts in image.
    """
    def __init__(self, args, input_filename, transforms, get_transform_params, img_key, caption_key, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}.')
        self.args = args
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets")
        df = pd.read_csv(os.path.join(self.root_path, input_filename), sep=sep, \
                        converters={"relation_phrase_list":ast.literal_eval, "attribute_phrase_list":ast.literal_eval, \
                                    "relation_attribute_phrase_bbox_list":ast.literal_eval, "neg_caption":ast.literal_eval, "neg_image":ast.literal_eval})
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.hard_captions = df["neg_caption"].tolist()
        self.relation_attribute_phrase_bbox_list = df["relation_attribute_phrase_bbox_list"].tolist()
        self.transforms = transforms
        self.get_transform_params = get_transform_params
        self.bbox_adapter = BBoxAdapter(224)
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def get_token_flag(self, original_bbox_list, transform_params):
        if self.args.model == "ViT-B-32":
            patch_size = 32
            image_token_num = 49
        elif self.args.model == "ViT-B-16":
            patch_size = 16
            image_token_num = 196
        elif self.args.model == "ViT-L-14":
            patch_size = 14
            image_token_num = 256

        if not original_bbox_list:
            mask_token_flag = torch.zeros(image_token_num + 1, dtype=torch.bool)
        else:
            transformed_bbox_list = [
                self.bbox_adapter.transform_bbox(b, transform_params) 
                for b in original_bbox_list
            ]
            mask_token_flag = bbox_to_patch_mask(transformed_bbox_list, image_size=224, patch_size=patch_size)

        return mask_token_flag

    def sample_part_region_mask(self, mask_token_flag, mask_region_reserve_ratio=0.8):
        # Initialize an all-False mask.
        new_mask = torch.zeros_like(mask_token_flag)
        # Get all True indices.
        true_indices = torch.where(mask_token_flag)[0]
        n = len(true_indices)
        if n == 0:
            return new_mask
        # Compute a rounded and clamped sample count.
        num_samples = int(round(n * mask_region_reserve_ratio))
        num_samples = max(0, min(num_samples, n))
        if num_samples > 0:
            # Randomly select indices.
            perm = torch.randperm(n)
            selected_indices = true_indices[perm[:num_samples]]
            new_mask[selected_indices] = True
        return new_mask
    
    def generate_random_mask(self, mask_ratio=0.6):
        if self.args.model == "ViT-B-32":
            image_token_num = 49
        elif self.args.model == "ViT-B-16":
            image_token_num = 196
        elif self.args.model == "ViT-L-14":
            image_token_num = 256
        mask = torch.zeros(image_token_num + 1, dtype=torch.bool)
        n = mask.numel() - 1  # Number of image tokens.
        num_mask = int(round(n * mask_ratio))
        num_mask = max(0, min(num_mask, n))  # Keep the value in range.
        
        if num_mask > 0:
            # Generate random indices and set them to True.
            indices = torch.randperm(n)[:num_mask]
            mask.view(-1)[indices + 1] = True
        return mask

    def __getitem__(self, idx):

        images = self.transforms(Image.open(os.path.join(self.data_path, str(self.images[idx]))))
        transform_params = self.get_transform_params()

        if self.relation_attribute_phrase_bbox_list[idx]:
            original_bbox_list = random.sample(list(self.relation_attribute_phrase_bbox_list[idx][0].values()), k=1)[0]
        else:
            original_bbox_list = []

        caption = self.captions[idx]
        caption = caption.strip()
        texts = tokenize([str(caption)])[0]
        
        try:
            chosen_caption = random.choice(self.hard_captions[idx])
            hard_captions = tokenize([str(chosen_caption)])[0] # hard negative caption for texts
        except:
            hard_captions = tokenize([str("A photo of full black image.")])[0]

        if self.args.random_mask_image:
            mask_token_flag = self.generate_random_mask(mask_ratio=self.args.random_mask_image_ratio)
        else:
            mask_token_flag = self.get_token_flag(original_bbox_list, transform_params)
            mask_token_flag = self.sample_part_region_mask(mask_token_flag, mask_region_reserve_ratio=self.args.mask_region_reserve_ratio)

        return images, texts, hard_captions, mask_token_flag
    
class CsvDataset_mask_text_image(Dataset):
    """
    Mask compositional concepts in image and text.
    """
    def __init__(self, args, input_filename, transforms, get_transform_params, img_key, caption_key, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}.')
        self.args = args
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets")
        df = pd.read_csv(os.path.join(self.root_path, input_filename), sep=sep, \
                        converters={"object_list":ast.literal_eval, "relation_list":ast.literal_eval,	"attribute_list":ast.literal_eval, \
                                    "relation_phrase_list":ast.literal_eval, "attribute_phrase_list":ast.literal_eval, "relation_attribute_phrase_bbox_list":ast.literal_eval, \
                                        "neg_caption":ast.literal_eval, "neg_image":ast.literal_eval})
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.hard_captions = df["neg_caption"].tolist()
        self.noun_list = df['object_list'].tolist()
        self.relation_word_list = df['relation_list'].tolist()
        self.adjective_word_list = df['attribute_list'].tolist()
        self.relation_attribute_phrase_bbox_list = df["relation_attribute_phrase_bbox_list"].tolist()
        self.transforms = transforms
        self.get_transform_params = get_transform_params
        self.bbox_adapter = BBoxAdapter(224)
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def get_word_token_pos_in_text_token(self, word, text):
        # get the word token position in all text token
        text_token_list = text.clone().tolist()
        word_token = _tokenizer.encode(word)
        start_idx = -1
        for idx in range(len(text)):
            if text_token_list[idx: idx + len(word_token)] == word_token:
                start_idx = idx
                break
        if start_idx != -1:
            return [start_idx + i for i in range(len(word_token))]
        else:
            # raise Exception("Not found matched word in the text!!!")
            return []

    def get_text_mask_token_flag(self, idx, texts):
        comp_concept_token_idx = [] # Indices of relation and attribute tokens in the full caption.
        object_token_idx = [] # Indices of object tokens in the full caption.
        attribute_word_list = self.adjective_word_list[idx]
        relation_word_list = self.relation_word_list[idx]
        object_word_list = self.noun_list[idx]
        
        for word in relation_word_list:
            # if word.lower() != "the" and word.lower() != "in" and word.lower() != "on":
            if word.lower() != "the":
                comp_concept_token_idx += self.get_word_token_pos_in_text_token(word, texts)
        
        for word in attribute_word_list:
            if word.lower() != "the":
                comp_concept_token_idx += self.get_word_token_pos_in_text_token(word, texts)

        for word in object_word_list:
            if word.lower() != "the":
                object_token_idx += self.get_word_token_pos_in_text_token(word, texts)
        
        comp_concept_token_flag = torch.zeros(len(texts)).bool()
        comp_concept_token_flag[comp_concept_token_idx] = True

        object_token_flag = torch.zeros(len(texts)).bool()
        object_token_flag[object_token_idx] = True

        return comp_concept_token_flag, object_token_flag

    def get_image_mask_token_flag(self, original_bbox_list, transform_params):
        if self.args.model == "ViT-B-32":
            patch_size = 32
            image_token_num = 49
        elif self.args.model == "ViT-B-16":
            patch_size = 16
            image_token_num = 196
        elif self.args.model == "ViT-L-14":
            patch_size = 14
            image_token_num = 256

        if not original_bbox_list:
            image_mask_token_flag = torch.zeros(image_token_num + 1, dtype=torch.bool)
        else:
            transformed_bbox_list = [
                self.bbox_adapter.transform_bbox(b, transform_params) 
                for b in original_bbox_list
            ]
            image_mask_token_flag = bbox_to_patch_mask(transformed_bbox_list, image_size=224, patch_size=patch_size)

        return image_mask_token_flag

    def generate_random_mask_image(self, mask_ratio=0.6):
        if self.args.model == "ViT-B-32":
            image_token_num = 49
        elif self.args.model == "ViT-B-16":
            image_token_num = 196
        elif self.args.model == "ViT-L-14":
            image_token_num = 256
        mask = torch.zeros(image_token_num + 1, dtype=torch.bool)
        n = mask.numel() - 1  # Number of image tokens.
        num_mask = int(round(n * mask_ratio))
        num_mask = max(0, min(num_mask, n))  # Keep the value in range.
        
        if num_mask > 0:
            # Generate random indices and set them to True.
            indices = torch.randperm(n)[:num_mask]
            mask.view(-1)[indices + 1] = True
        return mask

    def generate_random_rectangular_mask_image(self, mask_ratio=0.6):
        # Determine the grid size from the model type.
        if self.args.model == "ViT-B-32":
            grid_h, grid_w = 7, 7   # 7x7 grid.
        elif self.args.model == "ViT-B-16":
            grid_h, grid_w = 16, 16 # 16x16 grid.
        elif self.args.model == "ViT-L-14":
            grid_h, grid_w = 14, 14 # 14x14 grid.
        else:
            raise ValueError(f"Unsupported model: {self.args.model}")
        
        total_patches = grid_h * grid_w
        mask = torch.zeros(total_patches + 1, dtype=torch.bool)  # Include the class-token position.
        
        # Handle boundary cases.
        if mask_ratio <= 0:
            return mask
        elif mask_ratio >= 1:
            mask[1:] = True  # Mask all patches except the class token.
            return mask
        
        # Compute the rounded target mask area.
        target_area = int(round(total_patches * mask_ratio))
        target_area = max(1, min(target_area, total_patches))  # Keep the value in range.
        
        # Randomly generate rectangle parameters.
        while True:
            # Random rectangle height and width, at least one cell each.
            h = random.randint(1, grid_h)
            w = random.randint(1, grid_w)
            
            # Compute the current rectangle area.
            current_area = h * w
            
            # Accept rectangles close to the target area, with +/-25% tolerance.
            if abs(current_area - target_area) <= max(1, 0.25 * target_area):
                break
        
        # Randomly select the rectangle start position.
        top = random.randint(0, grid_h - h)
        left = random.randint(0, grid_w - w)
        
        # Create the grid mask.
        grid_mask = torch.zeros((grid_h, grid_w), dtype=torch.bool)
        # Set the rectangle region to True.
        grid_mask[top:top+h, left:left+w] = True
        
        # Flatten the grid mask and apply it to the output.
        mask[1:1+total_patches] = grid_mask.flatten()
        
        return mask

    def generate_random_mask_text(self, mask_ratio=0.15, texts=None):
        text_token_num = 77
        comp_concept_token_flag = torch.zeros(text_token_num, dtype=torch.bool)
        object_token_flag = torch.zeros(text_token_num, dtype=torch.bool)

        # Get the end-of-text token position. This path operates on a single sample.
        end_pos = texts.argmax(dim=-1).squeeze().item()  # Convert to a Python scalar.

        # Content tokens range from after the start token to before the end token.
        content_start = 1  # First position after the start-of-text token.
        content_end = end_pos  # End-of-text position, exclusive.

        # Valid content positions: [1, 2, ..., end_pos - 1].
        content_indices = torch.arange(content_start, content_end)
        n = len(content_indices)  # Number of content tokens.

        # Compute the number of tokens to mask.
        num_mask = int(round(n * mask_ratio))
        num_mask = max(0, min(num_mask, n))  # Keep the value in range.

        # Generate an independently sampled compositional-concept mask.
        if num_mask > 0:
            selected = torch.randperm(n)[:num_mask]  # Randomly select within the content range.
            comp_concept_token_flag[content_indices[selected]] = True

        # Generate an independently sampled object mask.
        if num_mask > 0:
            selected = torch.randperm(n)[:num_mask]  # Use a fresh random selection.
            object_token_flag[content_indices[selected]] = True

        return comp_concept_token_flag, object_token_flag

    def __getitem__(self, idx):

        images = self.transforms(Image.open(os.path.join(self.data_path, str(self.images[idx]))))
        transform_params = self.get_transform_params()

        if self.relation_attribute_phrase_bbox_list[idx]:
            original_bbox_list = random.sample(list(self.relation_attribute_phrase_bbox_list[idx][0].values()), k=1)[0]
        else:
            original_bbox_list = []

        caption = self.captions[idx]
        caption = caption.strip()
        texts = tokenize([str(caption)])[0]
        try:
            chosen_caption = random.choice(self.hard_captions[idx])
            hard_captions = tokenize([str(chosen_caption)])[0] # hard negative caption for texts
        except:
            chosen_caption = "A photo of full black image."
            hard_captions = tokenize([str(chosen_caption)])[0]

        comp_concept_token_flag, object_token_flag = self.get_text_mask_token_flag(idx, texts)       
        
        if random.random() < self.args.random_rectangular_mask_prob:
            image_mask_token_flag = self.generate_random_rectangular_mask_image(mask_ratio=self.args.random_rectangular_mask_image_ratio)
        else:
            image_mask_token_flag = self.get_image_mask_token_flag(original_bbox_list, transform_params)

        if self.args.random_mask_text:
            comp_concept_token_flag, object_token_flag = self.generate_random_mask_text(mask_ratio=self.args.random_mask_text_ratio, texts=texts)        
        
        if self.args.random_mask_image:
            if self.args.random_rectangular_mask_image:
                image_mask_token_flag = self.generate_random_rectangular_mask_image(mask_ratio=self.args.random_rectangular_mask_image_ratio)
            else:
                image_mask_token_flag = self.generate_random_mask_image(mask_ratio=self.args.random_mask_image_ratio)

        return images, texts, hard_captions, comp_concept_token_flag, object_token_flag, image_mask_token_flag



class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_txt(text):
    return tokenize([str(text)])[0]


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption(sample):
    return 'txt' in sample


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed() + epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed(self.worker_seed() + epoch)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png", text="txt"),
        wds.map_dict(image=preprocess_img, text=preprocess_txt),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, get_transform_params=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename

    if args.mask_text_image:
        dataset = CsvDataset_mask_text_image(
                args,
                input_filename,
                preprocess_fn,
                get_transform_params,
                img_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                sep=args.csv_separator)
    elif args.mask_image:
        dataset = CsvDataset_mask_image(
                args,
                input_filename,
                preprocess_fn,
                get_transform_params,
                img_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                sep=args.csv_separator)
    else:
        dataset = CsvDataset_mask_text(
                args,
                input_filename,
                preprocess_fn,
                img_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                sep=args.csv_separator)
                    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, get_transform_params=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        if args.mask_image or args.mask_text_image:
            data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_train, is_train=True, epoch=epoch, get_transform_params=get_transform_params)
        else:
            data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_train, is_train=True, epoch=epoch)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
