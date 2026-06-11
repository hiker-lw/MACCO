import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import ast
import json
import logging
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from easydict import EasyDict as edict

import open_clip

def evaluate_vl_checklist(pretrained_path, device):
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pretrained_path, device=device)
    model = model.eval()
    model = CLIPWrapper(model, device=device)
    # evaluate on VL_checklist Relation
    relation_score = {}
    relation_dataset_name_list = ["hake_action.json", "swig_action.json", "vg_action.json", "vg_spatial.json"]
    dataset_dir = "../datasets/VL_checklist"
    for dataset_name in relation_dataset_name_list:
        dataset = VL_checklist_Relation(image_preprocess=image_preprocess, root_dir=dataset_dir, relation_dataset=dataset_name)
        if dataset_name == "hake_action.json":
            loader = DataLoader(dataset, batch_size=1024, num_workers=20, shuffle=False)
        else:
            loader = DataLoader(dataset, batch_size=256, num_workers=20, shuffle=False)
        scores = model.get_retrieval_scores_batched(loader)
        avg_scores = dataset.evaluate_scores(scores)
        relation_score[dataset_name] = avg_scores
        print(f"-----------------------Evaluation on VL_checklist_Relation ({dataset_name}) Accuracy: {avg_scores:.4f}")

    print("*"*30)
    print(f"-----------------------Evaluation on VL_checklist_Relation (Action) Average Accuracy: {np.mean(list(relation_score.values())[:3]):.4f}")
    print(f"-----------------------Evaluation on VL_checklist_Relation (Spatial) Average Accuracy: {np.mean(list(relation_score.values())[3:]):.4f}")
    print(f"-----------------------Evaluation on VL_checklist_Relation Average Accuracy: {(np.mean(list(relation_score.values())[:3])+np.mean(list(relation_score.values())[3:]))/2:.4f}")
    print("*"*30)

    # evaluate on VL_checklist Attribute
    attribute_score = {}
    attribute_dataset_name_list = ["vaw/action.json", "vaw/color.json", "vaw/material.json", "vaw/size.json", "vaw/state.json", \
                        "vg/action.json", "vg/color.json", "vg/material.json", "vg/size.json", "vg/state.json"]
    dataset_dir = "../datasets/VL_checklist"
    for dataset_name in attribute_dataset_name_list:
        dataset = VL_checklist_Attribute(image_preprocess=image_preprocess, root_dir=dataset_dir, attribute_dataset=dataset_name)
        if dataset_name == "vaw/color.json":
            loader = DataLoader(dataset, batch_size=1024, num_workers=20, shuffle=False)
        else:
            loader = DataLoader(dataset, batch_size=256, num_workers=20, shuffle=False)
        scores = model.get_retrieval_scores_batched(loader)
        avg_scores = dataset.evaluate_scores(scores)
        attribute_score[dataset_name] = avg_scores
        print(f"-----------------------Evaluation on VL_checklist_Attribute ({dataset_name}) Accuracy: {avg_scores:.4f}")

    attribute_score_dict = {"action": [], "color": [], "material": [], "size": [], "state": []}
    for key in attribute_score.keys():
        for type in ["action", "color", "material", "size", "state"]:
            if type in key:
                attribute_score_dict[type].append(attribute_score[key])

    for key in attribute_score_dict:
        attribute_score_dict[key] = np.mean(attribute_score_dict[key])

    print("*"*30)
    print(f"-----------------------Evaluation on VL_checklist_Attribute ({attribute_score_dict})")
    print(f"-----------------------Evaluation on VL_checklist_Attribute Average Accuracy: {np.mean(list(attribute_score_dict.values())):.4f}")
    print("*"*30)


    # evaluate on VL_checklist Object
    object_score = {"Location": {"margin": {}, "center": {}, "mid": {}}, "Size": {"large": {}, "medium": {}, "small": {}}}
    dataset_dir = "../MACCO_CLIP/datasets/VL_checklist"
    for object_eval_type in object_score.keys():
        for value in object_score[object_eval_type].keys():
            dataset_name_list = os.listdir(os.path.join(dataset_dir, f"VL_checklist_json_data/Object/{object_eval_type}/{value}"))
            for dataset_name in dataset_name_list:
                dataset = VL_checklist_Object(image_preprocess=image_preprocess, root_dir=dataset_dir, \
                                              dataset_name=f"{object_eval_type}/{value}/{dataset_name}", add_mark=False)
                loader = DataLoader(dataset, batch_size=256, num_workers=20, shuffle=False)
                scores = model.get_retrieval_scores_batched(loader)
                avg_scores = dataset.evaluate_scores(scores)
                object_score[object_eval_type][value][dataset_name] = avg_scores
                print(f"-----------------------Evaluation on VL_checklist_Object ({object_eval_type}-{value}-{dataset_name}) Accuracy: {avg_scores:.4f}")
    
    object_score_avg = {"Location": {}, "Size": {}}
    for object_eval_type in object_score.keys():
        for value in object_score[object_eval_type].keys():
            object_score_avg[object_eval_type][value] = np.round(np.mean(list(object_score[object_eval_type][value].values())), 4)
    
    print("*"*30)
    print(f"-----------------------Evaluation on VL_checklist_Object Detailed Accuracy: {object_score_avg}")
    print(f"-----------------------Evaluation on VL_checklist_Object (Location) Average Accuracy: {np.mean(list(object_score_avg['Location'].values())):.4f}")
    print(f"-----------------------Evaluation on VL_checklist_Object (Size) Average Accuracy: {np.mean(list(object_score_avg['Size'].values())):.4f}")
    print(f"-----------------------Evaluation on VL_checklist_Object Average Accuracy: {(np.mean(list(object_score_avg['Location'].values()))+np.mean(list(object_score_avg['Size'].values())))/2:.4f}")
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

class VL_checklist_Relation(Dataset):
    def __init__(self, image_preprocess, root_dir, relation_dataset):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        root_dir: Directory for the VL-checklist dataset.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "VL_checklist_json_data", "Relation", relation_dataset)
        image_dir = os.path.join(root_dir, "VL_checklist_datasets", relation_dataset.split("_")[0])

        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)
        
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["POS"]
        false_caption = test_case["NEG"]
        
        item = edict({"image_options": [image], "caption_options": [false_caption, true_caption]})
        return item
        
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 1)
        avg_scores = np.mean(correct_mask)
        return avg_scores

class VL_checklist_Attribute(Dataset):
    def __init__(self, image_preprocess, root_dir, attribute_dataset):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        root_dir: Directory for the VL-checklist dataset.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "VL_checklist_json_data", "Attribute", attribute_dataset)
        image_dir = os.path.join(root_dir, "VL_checklist_datasets", "vg")

        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)
        
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["POS"]
        false_caption = test_case["NEG"]
        
        item = edict({"image_options": [image], "caption_options": [false_caption, true_caption]})
        return item
        
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 1)
        avg_scores = np.mean(correct_mask)
        return avg_scores

class VL_checklist_Object(Dataset):
    def __init__(self, image_preprocess, root_dir, dataset_name, add_mark):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        root_dir: Directory for the VL-checklist dataset.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "VL_checklist_json_data/Object", dataset_name)
        image_dir = os.path.join(root_dir, "VL_checklist_datasets", os.path.basename(dataset_name).split("_")[0])

        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)
        
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])

        self.image_preprocess = image_preprocess
        self.add_mark = add_mark

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["POS"]
        false_caption = test_case["NEG"]
        
        item = edict({"image_options": [image], "caption_options": [false_caption, true_caption]})
        return item
        
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 1)
        avg_scores = np.mean(correct_mask)
        return avg_scores
    
if __name__ == '__main__':
    pretrained_path = "/your_ckpt_path/xxx.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_vl_checklist(pretrained_path=pretrained_path, device=device)