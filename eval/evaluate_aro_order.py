import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
from easydict import EasyDict as edict
import subprocess

from ARO.dataset_zoo.perturbations import TextShuffler
from ARO.dataset_zoo.constants import ARO_ROOT, COCO_ROOT, FLICKR_ROOT
from ARO.dataset_zoo.retrieval import pre_caption

import open_clip

def evaluate_aro_coco_order(pretrained_path, device):
    print("*"*30)
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pretrained_path, device=device)
    model = model.eval()
    model = CLIPWrapper(model, device=device)

    # evaluate on COCO_Order
    dataset_dir = "../datasets/ARO"
    coco_order_dataset = COCO_Order(image_preprocess=image_preprocess, root_dir=dataset_dir)
    coco_order_loader = DataLoader(coco_order_dataset, batch_size=256, num_workers=20, shuffle=False)
    coco_order_scores = model.get_retrieval_scores_batched(coco_order_loader)
    coco_order_records = coco_order_dataset.evaluate_scores(coco_order_scores)[0]["Precision@1"]
    print(f"COCO_Order Accuracy: {coco_order_records:.3f}")
    print("*"*30)

def evaluate_aro_flickr30k_order(pretrained_path, device):
    print("*"*30)
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pretrained_path, device=device)
    model = model.eval()
    model = CLIPWrapper(model, device=device)

    # evaluate on Flickr30k_Order
    dataset_dir = "../datasets/ARO"
    flickr30k_order_dataset = Flickr30k_Order(image_preprocess=image_preprocess, root_dir=dataset_dir)
    flickr30k_order_loader = DataLoader(flickr30k_order_dataset, batch_size=256, num_workers=20, shuffle=False)
    flickr30k_order_scores = model.get_retrieval_scores_batched(flickr30k_order_loader)
    flickr30k_order_records = flickr30k_order_dataset.evaluate_scores(flickr30k_order_scores)[0]["Precision@1"]
    print(f"Flickr30k_Order Accuracy: {flickr30k_order_records:.3f}")
    print("*"*30)



class CLIPWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions, "focused_snippets_flag" is a list of focused snippet position flag.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")
        for batch in tqdm_loader:
            image_options = []
            for i_option in batch["image_options"]:
                image_embeddings = self.model.encode_image(i_option.to(self.device)).cpu().numpy() # B x D
                image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True) # B x D
                image_options.append(np.expand_dims(image_embeddings, axis=1))
            
            caption_options = []
            for c_option in batch["caption_options"]:
                caption_tokenized = torch.cat([open_clip.tokenize(c) for c in c_option])
                caption_embeddings = self.model.encode_text(caption_tokenized.to(self.device)).cpu().numpy() # B x D
                caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True) # B x D
                caption_options.append(np.expand_dims(caption_embeddings, axis=1))
                
            image_options = np.concatenate(image_options, axis=1) # B x K x D
            caption_options = np.concatenate(caption_options, axis=1) # B x L x D
            batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options) # B x K x L
            scores.append(batch_scores)
        
        all_scores = np.concatenate(scores, axis=0) # N x K x L
        return all_scores

class COCO_Order(Dataset):
    def __init__(self, image_preprocess=None, root_dir=COCO_ROOT, max_words=30, split="test",
                 image_perturb_fn=None, download=False):  
        """
        COCO Order Dataset.
        image_preprocess: image preprocessing function
        root_dir: The directory of the coco dataset. This directory should contain test2014 files.
        max_words: Cropping the caption to max_words.
        split: 'val' or 'test'
        image_perturb_fn: not used; for compatibility.
        download: Whether to download the dataset if it does not exist.
        """
        shuffler = TextShuffler()
        perturb_functions = [shuffler.shuffle_nouns_and_adj, shuffler.shuffle_allbut_nouns_and_adj,
                             shuffler.shuffle_within_trigrams, shuffler.shuffle_trigrams]

        self.root_dir = root_dir

        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}        
        self.annotation = json.load(open(os.path.join(root_dir,filenames[split]),'r'))
        self.image_preprocess = image_preprocess
        self.image_root = os.path.join(os.path.dirname(root_dir), "coco")
        
        self.test_cases = []
        
        for img_id, ann in tqdm(enumerate(self.annotation)):
            for i, caption in enumerate(ann['caption']):
                test_case = {}
                test_case["image"] = ann["image"]
                test_case["caption_options"] = [pre_caption(caption,max_words)]

                for perturb_fn in perturb_functions:
                    test_case["caption_options"].append(pre_caption(perturb_fn(caption), max_words))
                self.test_cases.append(test_case)
                                    
    def __len__(self):
        return len(self.test_cases)
    
    def __getitem__(self, index):  
        test_case = self.test_cases[index]  
        image_path = os.path.join(self.image_root, test_case["image"])       
         
        image = Image.open(image_path).convert('RGB')    
        if self.image_preprocess is not None: 
            image = self.image_preprocess(image)  
        
        item = edict({"image_options": [image], "caption_options": test_case["caption_options"]})
        return item
        
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        
        else:
            scores_t2i = scores
            scores_i2t = scores
        
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        records = [{"Precision@1": np.mean(correct_mask)}]
        return records


class Flickr30k_Order(Dataset):
    def __init__(self, image_preprocess, split="test", root_dir=FLICKR_ROOT, max_words=30,
                 *args, **kwargs):  
        """
        image_preprocess: image preprocessing function
        split: 'val' or 'test'
        root_dir: The directory of the flickr30k images. This should contain the `flickr30k-images` directory that \
            contains all the images. 
        """
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        self.annotation = json.load(open(os.path.join(root_dir,filenames[split]),'r'))
        self.image_preprocess = image_preprocess
        self.root_dir = root_dir
        
        self.test_cases = []
        
        shuffler = TextShuffler()
        perturb_functions = [shuffler.shuffle_nouns_and_adj, shuffler.shuffle_allbut_nouns_and_adj,
                             shuffler.shuffle_within_trigrams, shuffler.shuffle_trigrams]
        for img_id, ann in tqdm(enumerate(self.annotation)):
            for i, caption in enumerate(ann['caption']):
                test_case = {}
                test_case["image"] = ann["image"]
                test_case["caption_options"] = [pre_caption(caption,max_words)]

                for perturb_fn in perturb_functions:
                    test_case["caption_options"].append(pre_caption(perturb_fn(caption), max_words))
                self.test_cases.append(test_case)
                                
    def __len__(self):
        return len(self.test_cases)
    
    def __getitem__(self, index):  
        test_case = self.test_cases[index]  
        image_path = os.path.join(self.root_dir, "flickr30k-images", test_case["image"])        
        image = Image.open(image_path).convert('RGB')    
        
        if self.image_preprocess is not None: 
            image = self.image_preprocess(image)  
            
        item = edict({"image_options": [image], "caption_options": test_case["caption_options"]})
        return item
    
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        else:
            scores_t2i = scores
            scores_i2t = scores
        
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        result_records = [{"Precision@1": np.mean(correct_mask)}]
        return result_records


if __name__ == '__main__':
    pretrained_path = "/your_ckpt_path/xxx.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_aro_coco_order(pretrained_path, device)
    evaluate_aro_flickr30k_order(pretrained_path, device)