import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import json
import logging
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
from easydict import EasyDict as edict

import open_clip

def evaluate_aro_relation(pretrained_path, device):
    print("*"*30)
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pretrained_path, device=device)
    model = model.eval()
    model = CLIPWrapper(model, device=device)

    # evaluate on ARO Relation
    dataset_dir = "../datasets/ARO"
    vgr_dataset = VG_Relation(image_preprocess=image_preprocess, root_dir=dataset_dir)
    vgr_loader = DataLoader(vgr_dataset, batch_size=256, num_workers=20, shuffle=False)
    vgr_scores = model.get_retrieval_scores_batched(vgr_loader)
    vgr_records = vgr_dataset.evaluate_scores(vgr_scores)
    symmetric = ['leaning against','pulled by','pulling','adjusting', 'attached to', 'between', 'bigger than', 'biting', 'boarding', 'brushing', 'chewing', 'cleaning', 'climbing', 'close to', 'coming from', 'coming out of', 'contain', 'crossing', 'dragging', 'draped over', 'drinking', 'drinking from', 'driving', 'driving down', 'driving on', 'eating from', 'eating in', 'enclosing', 'exiting', 'facing', 'filled with', 'floating in', 'floating on', 'flying', 'flying above', 'flying in', 'flying over', 'flying through', 'full of', 'going down', 'going into', 'going through', 'grazing in', 'growing in', 'growing on', 'guiding', 'hanging from', 'hanging in', 'hanging off', 'hanging over', 'higher than', 'holding onto', 'hugging', 'in between', 'jumping off', 'jumping on', 'jumping over', 'kept in', 'larger than', 'leading', 'leaning over', 'leaving', 'licking', 'longer than', 'looking in', 'looking into', 'looking out', 'looking over', 'looking through', 'lying next to', 'lying on top of', 'making', 'mixed with', 'mounted on', 'moving', 'on the back of', 'on the edge of', 'on the front of', 'on the other side of', 'opening', 'painted on', 'parked at', 'parked beside', 'parked by', 'parked in', 'parked in front of', 'parked near', 'parked next to', 'perched on', 'petting', 'piled on', 'playing', 'playing in', 'playing on', 'playing with', 'pouring', 'reaching for', 'reading', 'reflected on', 'riding on', 'running in', 'running on', 'running through', 'seen through', 'sitting behind', 'sitting beside', 'sitting by', 'sitting in front of', 'sitting near', 'sitting next to', 'sitting under', 'skiing down', 'skiing on', 'sleeping in', 'sleeping on', 'smiling at', 'sniffing', 'splashing', 'sprinkled on', 'stacked on', 'standing against', 'standing around', 'standing behind', 'standing beside', 'standing in front of', 'standing near', 'standing next to', 'staring at', 'stuck in', 'surrounding', 'swimming in', 'swinging', 'talking to', 'topped with', 'touching', 'traveling down', 'traveling on', 'tying', 'typing on', 'underneath', 'wading in', 'waiting for', 'walking across', 'walking by', 'walking down', 'walking next to', 'walking through', 'working in', 'working on', 'worn on', 'wrapped around', 'wrapped in', 'by', 'of', 'near', 'next to', 'with', 'beside', 'on the side of', 'around']
    df = pd.DataFrame(vgr_records)
    df = df[~df.Relation.isin(symmetric)]
    df = df.round({'Accuracy': 4})
    print(f"VG-Relation Macro Accuracy: {df.Accuracy.mean():.3f}")
    print("*"*30)


def evaluate_aro_attribute(pretrained_path, device):
    print("*"*30)
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pretrained_path, device=device)
    model = model.eval()
    model = CLIPWrapper(model, device=device)

    # evaluate on ARO Attribute
    dataset_dir = "../datasets/ARO_Relation_dataset"
    vga_dataset = VG_Attribution(image_preprocess=image_preprocess, root_dir=dataset_dir)
    vga_loader = DataLoader(vga_dataset, batch_size=256, num_workers=20, shuffle=False)
    vga_scores = model.get_retrieval_scores_batched(vga_loader)
    vga_records = vga_dataset.evaluate_scores(vga_scores)
    df = pd.DataFrame(vga_records)
    print(df)
    df = df.round({'Accuracy': 4})
    print(f"VG-Attribution Macro Accuracy: {df.Accuracy.mean():.3f}")
    print("*"*30)

def evaluate_winoground(pretrained_path, device):
    print("*"*30)
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pretrained_path, device=device)
    model = model.eval()
    model = CLIPWrapper(model, device=device)

    # evaluate on winoground full version
    dataset_dir = "../datasets/winoground"
    wino_dataset = Winoground(image_preprocess=image_preprocess, root_dir=dataset_dir, just_test_relation=False, just_test_clean=False)
    wino_loader = DataLoader(wino_dataset, batch_size=32, num_workers=20, shuffle=False)
    # Compute the scores for each test case
    wino_scores = model.get_retrieval_scores_batched(wino_loader)
    text_score, image_score, group_score = wino_dataset.evaluate_scores(wino_scores)
    print(f"Accuracy on winoground full version, text_score: {text_score:.3f}, image_score: {image_score:.3f}, group_score: {group_score:.3f}")

    # evaluate on winoground clean version
    dataset_dir = "../datasets/winoground"
    wino_dataset = Winoground(image_preprocess=image_preprocess, root_dir=dataset_dir, just_test_relation=False, just_test_clean=True)
    wino_loader = DataLoader(wino_dataset, batch_size=32, num_workers=20, shuffle=False)
    # Compute the scores for each test case
    wino_scores = model.get_retrieval_scores_batched(wino_loader)
    text_score, image_score, group_score = wino_dataset.evaluate_scores(wino_scores)
    print(f"Accuracy on winoground clean version, text_score: {text_score:.3f}, image_score: {image_score:.3f}, group_score: {group_score:.3f}")
    print("*"*30)

def evaluate_sugar_crepe(pretrained_path, device):
    print("*"*30)
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pretrained_path, device=device)
    model = model.eval()
    model = CLIPWrapper(model, device=device)

    # evaluate on sugar_crepe
    relation_score = {}
    relation_dataset_name_list = ["add_att.json", "replace_att.json", "swap_att.json", "replace_rel.json", \
                                  "add_obj.json", "replace_obj.json", "swap_obj.json"]
    dataset_dir = "../datasets/sugar-crepe"
    for dataset_name in relation_dataset_name_list:
        dataset = SugarCrepe(image_preprocess=image_preprocess, root_dir=dataset_dir, dataset_name=dataset_name)
        loader = DataLoader(dataset, batch_size=256, num_workers=8, shuffle=False)
        scores = model.get_retrieval_scores_batched(loader)
        avg_scores = dataset.evaluate_scores(scores)
        relation_score[dataset_name] = avg_scores
        print(f"-----------------------Evaluation on sugar_crepe ({dataset_name}) Accuracy: {avg_scores:.4f}")
    
    print(f"-----------------------Evaluation on sugar_crepe Attribute Average Accuracy: {np.mean(list(relation_score.values())[:3]):.4f}")
    print(f"-----------------------Evaluation on sugar_crepe Relation Average Accuracy: {np.mean(list(relation_score.values())[3:4]):.4f}")
    print(f"-----------------------Evaluation on sugar_crepe Object Average Accuracy: {np.mean(list(relation_score.values())[4:]):.4f}")
    print("*"*30)

def evaluate_valse(pretrained_path, device):
    print("*"*30)
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pretrained_path, device=device)
    model = model.eval()
    model = CLIPWrapper(model, device=device)

    # evaluate on VALSE
    score = {}
    dataset_name_list = ["actant-swap.json", "action-replacement.json", "coreference-hard.json", "coreference-standard.json", \
                                  "counting-adversarial.json", "counting-hard.json", "counting-small-quant.json", \
                                    "existence.json", "foil-it.json", "plurals.json", "relations.json"]
    
    for dataset_name in dataset_name_list:
        dataset_dir = "../datasets"
        if dataset_name in ["actant-swap.json", "action-replacement.json"]:
            dataset_dir = os.path.join(dataset_dir, "VL_checklist/VL_checklist_datasets")
        elif dataset_name in ["coreference-hard.json", "counting-adversarial.json", "counting-hard.json", "counting-small-quant.json", "existence.json"]:
            dataset_dir = os.path.join(dataset_dir, "VALSE")
        elif dataset_name in ["coreference-standard.json", "foil-it.json"]:
            dataset_dir = os.path.join(dataset_dir, "coco")
        elif dataset_name in ["plurals.json", "relations.json"]:
            dataset_dir = os.path.join(dataset_dir, "sugar-crepe")

        dataset = VALSE(image_preprocess=image_preprocess, root_dir=dataset_dir, dataset_name=dataset_name)
        loader = DataLoader(dataset, batch_size=256, num_workers=8, shuffle=False)
        scores = model.get_retrieval_scores_batched(loader)
        avg_scores = dataset.evaluate_scores(scores)
        score[dataset_name] = avg_scores
        print(f"-----------------------Evaluation on VALSE ({dataset_name}) Accuracy: {avg_scores:.4f}")

    print(f"\n-----------------------Evaluation on VALSE <Action Relation> Average Accuracy: {np.mean(list(score.values())[:2]):.4f}")
    print(f"-----------------------Evaluation on VALSE <Spatial Relation> Average Accuracy: {np.mean(list(score.values())[-1:]):.4f}")
    print(f"-----------------------Evaluation on VALSE <Relation> Average Accuracy: {(np.mean(list(score.values())[:2])+np.mean(list(score.values())[-1:]))/2:.4f}")
    print("*"*30)

def evaluate_whats_up(pretrained_path, device):
    print("*"*30)
    model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pretrained_path, device=device)
    model = model.eval()
    model = CLIPWrapper(model, device=device)

    dataset_dir = "/home/liwei/code/MACCO_CLIP/datasets/whats_up_data"
    # evaluate on Controlled_Images
    print("*"*30)
    whats_up_dataset = Controlled_Images(image_preprocess=image_preprocess, root_dir=dataset_dir, subset='A')
    whats_up_loader = DataLoader(whats_up_dataset, batch_size=256, num_workers=8, shuffle=False)
    whats_up_scores = model.get_retrieval_scores_batched(whats_up_loader)
    whats_up_records, individual_accuracy_1 = whats_up_dataset.evaluate_scores(whats_up_scores)
    # print(f"Fine-grained accuracy on What's up subset A: {whats_up_records}")
    # df = pd.DataFrame(whats_up_records)
    # df = df.round({'Accuracy': 4})
    # print(f"What's up subset A Macro Accuracy: {df.Accuracy.mean():.3f}")
    print(f"What's up subset A Accuracy: {individual_accuracy_1:.3f}")


    whats_up_dataset = Controlled_Images(image_preprocess=image_preprocess, root_dir=dataset_dir, subset='B')
    whats_up_loader = DataLoader(whats_up_dataset, batch_size=256, num_workers=8, shuffle=False)
    whats_up_scores = model.get_retrieval_scores_batched(whats_up_loader)
    whats_up_records, individual_accuracy_2 = whats_up_dataset.evaluate_scores(whats_up_scores)
    # print(f"Fine-grained accuracy on What's up subset B: {whats_up_records}")
    # df = pd.DataFrame(whats_up_records)
    # df = df.round({'Accuracy': 4})
    # print(f"What's up subset B Macro Accuracy: {df.Accuracy.mean():.3f}")
    print(f"What's up subset B Accuracy: {individual_accuracy_2:.3f}")
    print(f"What's up Macro Accuracy: {(individual_accuracy_1 + individual_accuracy_2)/2:.3f}")
    print("*"*30)

    # evaluate on COCO_QA
    print("*"*30)
    coco_qa_dataset = COCO_QA(image_preprocess=image_preprocess, root_dir=dataset_dir, subset='one')
    coco_qa_loader = DataLoader(coco_qa_dataset, batch_size=256, num_workers=20, shuffle=False)
    coco_qa_scores = model.get_retrieval_scores_batched(coco_qa_loader)
    coco_qa_records, individual_accuracy_1 = coco_qa_dataset.evaluate_scores(coco_qa_scores)
    # print(f"Fine-grained accuracy on COCO_QA subset one: {coco_qa_records}")
    # df = pd.DataFrame(coco_qa_records)
    # df = df.round({'Accuracy': 4})
    # print(f"COCO_QA subset one Macro Accuracy: {df.Accuracy.mean():.3f}")
    print(f"COCO_QA subset one Accuracy: {individual_accuracy_1:.3f}")

    coco_qa_dataset = COCO_QA(image_preprocess=image_preprocess, root_dir=dataset_dir, subset='two')
    coco_qa_loader = DataLoader(coco_qa_dataset, batch_size=256, num_workers=20, shuffle=False)
    coco_qa_scores = model.get_retrieval_scores_batched(coco_qa_loader)
    coco_qa_records, individual_accuracy_2 = coco_qa_dataset.evaluate_scores(coco_qa_scores)
    # print(f"Fine-grained accuracy on COCO_QA subset two: {coco_qa_records}")
    # df = pd.DataFrame(coco_qa_records)
    # df = df.round({'Accuracy': 4})
    # print(f"COCO_QA subset two Macro Accuracy: {df.Accuracy.mean():.3f}")
    print(f"COCO_QA subset one Accuracy: {individual_accuracy_2:.3f}")
    print(f"COCO_QA Macro Accuracy: {(individual_accuracy_1 + individual_accuracy_2)/2:.3f}")
    print("*"*30)

    # evaluate on VG_QA
    print("*"*30)
    vg_qa_dataset = VG_QA(image_preprocess=image_preprocess, root_dir=dataset_dir, subset='one')
    vg_qa_loader = DataLoader(vg_qa_dataset, batch_size=256, num_workers=20, shuffle=False)
    vg_qa_scores = model.get_retrieval_scores_batched(vg_qa_loader)
    vg_qa_records, individual_accuracy_1 = vg_qa_dataset.evaluate_scores(vg_qa_scores)
    # print(f"Fine-grained accuracy on VG_QA subset one: {vg_qa_records}")
    # df = pd.DataFrame(vg_qa_records)
    # df = df.round({'Accuracy': 4})
    # print(f"VG_QA subset one Macro Accuracy: {df.Accuracy.mean():.3f}")
    print(f"VG_QA subset one Accuracy: {individual_accuracy_1:.3f}")

    vg_qa_dataset = COCO_QA(image_preprocess=image_preprocess, root_dir=dataset_dir, subset='two')
    vg_qa_loader = DataLoader(vg_qa_dataset, batch_size=256, num_workers=20, shuffle=False)
    vg_qa_scores = model.get_retrieval_scores_batched(vg_qa_loader)
    vg_qa_records, individual_accuracy_2 = vg_qa_dataset.evaluate_scores(vg_qa_scores)
    # print(f"Fine-grained accuracy on VG_QA subset two: {vg_qa_records}")
    # df = pd.DataFrame(vg_qa_records)
    # df = df.round({'Accuracy': 4})
    # print(f"VG_QA subset two Macro Accuracy: {df.Accuracy.mean():.3f}")
    print(f"VG_QA subset one Accuracy: {individual_accuracy_2:.3f}")
    print(f"VG_QA Macro Accuracy: {(individual_accuracy_1 + individual_accuracy_2)/2:.3f}")
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
    
    
class VG_Relation(Dataset):
    def __init__(self, image_preprocess, root_dir):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        root_dir: Directory for the VG-R dataset.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "visual_genome_relation.json")
        image_dir = os.path.join(root_dir, "images")

        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)
        
        self.all_relations = list()
        self.all_image_path =list()
        self.all_true_caption =list()
        self.all_false_caption =list()
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])
            self.all_image_path.append(item["image_path"])
            self.all_true_caption.append(item["true_caption"])
            self.all_false_caption.append(item["false_caption"])

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"], test_case["bbox_y"] + test_case["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
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

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 1)
        metrics["Accuracy"] = np.mean(correct_mask)

        all_relations = np.array(self.all_relations)

        result_records = []
        # Log the accuracy of all relations
        for relation in np.unique(all_relations):
            relation_mask = (all_relations == relation)
            if relation_mask.sum() == 0:
                continue
            result_records.append({
                "Relation": relation,
                "Accuracy": correct_mask[relation_mask].mean(),
                "Count": relation_mask.sum(),
                "Dataset": "Visual Genome Relation"
            })
        return result_records

class VG_Attribution(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir="", download=False):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        text_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        image_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        root_dir: Directory for the VG-A dataset.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "visual_genome_attribution.json")
        image_dir = os.path.join(root_dir, "images")
        
        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)
        
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
        
        # Set of attributes in each test case
        self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"], test_case["bbox_y"] + test_case["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
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
        result_records = []
        all_attributes = np.array(self.all_attributes)
        for attr in np.unique(all_attributes):
            attr_mask = (all_attributes == attr)
            if attr_mask.sum() < 25:
                continue
            result_records.append({
                "Attributes": attr,
                "Accuracy": correct_mask[attr_mask].mean(),
                "Count": attr_mask.sum(),
                "Dataset": "Visual Genome Attribution"
            })
        return result_records

class Winoground(Dataset):
    def __init__(self, image_preprocess, root_dir, just_test_relation: bool, just_test_clean: bool):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        root_dir: Directory for the Winoground Relation dataset.
        '''
        self.root_dir = root_dir
        
        clean_index = [0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 20, 21, 24, 26, 29, 30, 32, 33, 34, 35, 37, 39, 43, 45, 47, 48, 50, 51, 52, 53, 54, 56, 57, 59, 60, 64, 66, 67, 71, 79, 80, 85, 87, 89, 90, 91, 92, 94, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 112, 115, 117, 120, 122, 123, 124, 125, 126, 127, 129, 137, 139, 140, 141, 142, 145, 146, 147, 151, 153, 154, 157, 158, 160, 161, 162, 165, 166, 167, 168, 169, 170, 171, 175, 177, 178, 179, 180, 181, 183, 184, 185, 186, 194, 195, 196, 197, 202, 205, 207, 212, 213, 216, 225, 231, 236, 240, 243, 244, 248, 250, 251, 252, 256, 259, 261, 265, 266, 269, 270, 271, 272, 273, 278, 279, 283, 285, 288, 289, 290, 291, 294, 297, 301, 302, 306, 308, 309, 317, 328, 337, 341, 349, 357, 360, 366, 368, 369, 370, 372, 378, 379, 380, 389, 391, 397]


        if just_test_relation:
            # just test samples whose tag is "Relation"
            annotation_file = os.path.join(root_dir, "winoground_relation.json")
        else:
            # test all samples
            annotation_file = os.path.join(root_dir, "examples.jsonl")

        image_dir = os.path.join(root_dir, "images")
        if not os.path.exists(image_dir):
            print("Image Directory for Winoground could not be found!")
        
        if not os.path.exists(annotation_file):
            print("Annotation file for Winoground could not be found!") 
        
        self.dataset = []
        if just_test_relation:
            with open(annotation_file, "r") as f:
                full_dataset_relation = json.load(f)
            if just_test_clean:
                self.dataset = [item for _, item in enumerate(full_dataset_relation) if item["id"] in clean_index]
            else:
                self.dataset = full_dataset_relation
                
        else:
            with open(annotation_file, "r") as f:
                full_dataset = [json.loads(line) for line in f]
            if just_test_clean:
                self.dataset = [item for _, item in enumerate(full_dataset) if item["id"] in clean_index]
            else:
                self.dataset = full_dataset
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image_0 = Image.open(os.path.join(self.root_dir, "images", test_case["image_0"] + ".png")).convert('RGB')
        image_1 = Image.open(os.path.join(self.root_dir, "images", test_case["image_1"] + ".png")).convert('RGB')


        if self.image_preprocess is not None:
            image_0 = self.image_preprocess(image_0)
            image_1 = self.image_preprocess(image_1)

        # Each test case has a correct and incorrect caption.
        caption_0 = test_case["caption_0"]
        caption_1 = test_case["caption_1"]
        item = edict({"image_options": [image_0, image_1], "caption_options": [caption_0, caption_1]})
        return item
    
    def evaluate_scores(self, scores):
        """
        Scores: N x 2 x 2
        """
        N = scores.shape[0]
        text_score = np.zeros(N, dtype=bool)
        image_score = np.zeros(N, dtype=bool)

        for i in range(N):
            text_score[i] = scores[i,0,0] > scores[i,0,1] and scores[i,1,1] > scores[i,1,0]
            image_score[i] = scores[i,0,0] > scores[i,1,0] and scores[i,1,1] > scores[i,0,1]
        group_score = np.logical_and(text_score, image_score)
        return text_score.sum()/N, image_score.sum()/N, group_score.sum()/N

class SugarCrepe(Dataset):
    def __init__(self, image_preprocess, root_dir, dataset_name):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        root_dir: Directory for the sugar_crepe dataset.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, dataset_name)
        image_dir = os.path.join(root_dir, "val2017")

        with open(annotation_file, "r") as f:
            self.dataset = list(json.load(f).values())
        
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["filename"])

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["caption"]
        false_caption = test_case["negative_caption"]
        
        
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

class VALSE(Dataset):
    def __init__(self, image_preprocess, root_dir, dataset_name):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        root_dir: Directory for the sugar_crepe dataset.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join("../MACCO_CLIP/datasets/VALSE/probing_data", dataset_name)

        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)
        
        for item in self.dataset:
            item["image_path"] = os.path.join(self.root_dir, item["image_path"])

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["pos_caption"]
        false_caption = test_case["neg_caption"]
        
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


class Controlled_Images(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir="", download=False, subset='A'):
        self.root_dir = root_dir
        if subset == 'A':
            annotation_file = os.path.join(root_dir, "controlled_images_dataset.json")
            image_dir = os.path.join(root_dir, 'controlled_images')
        else:
            annotation_file = os.path.join(root_dir, "controlled_clevr_dataset.json")
            image_dir = os.path.join(root_dir, 'controlled_clevr')

        self.dataset = json.load(open(annotation_file))
        self.subset = subset
        self.all_prepositions = []
        if self.subset == 'A':
            for d in self.dataset:
                if 'left_of' in d['image_path']:
                    self.all_prepositions.append('left_of')
                elif 'right_of' in d['image_path']:
                    self.all_prepositions.append('right_of')
                elif '_on_' in d['image_path']:
                    self.all_prepositions.append('on')
                else:
                    self.all_prepositions.append('under')
            self.eval_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': 0, 'right': 0, \
                                'on': 0, 'under': 0} for d in self.dataset}
            self.pred_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': '', 'right': '', \
                                'on': '', 'under': ''} for d in self.dataset}


        else:
            for d in self.dataset:
                if 'left_of' in d['image_path']:
                    self.all_prepositions.append('left_of')
                elif 'right_of' in d['image_path']:
                    self.all_prepositions.append('right_of')
                elif '_in-front_of_' in d['image_path']:
                    self.all_prepositions.append('in-front_of')
                else:
                    self.all_prepositions.append('behind')
            self.eval_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': 0, 'right': 0, \
                                'in-front': 0, 'behind': 0} for d in self.dataset}
            self.pred_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': '', 'right': '', \
                                'in-front': '', 'behind': ''} for d in self.dataset}

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(os.path.join(self.root_dir, test_case["image_path"])).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        
        item = edict({"image_options": [image], "caption_options": test_case['caption_options']})
        return item


    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 4, i.e. first caption is right, next three captions are wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        metrics["Accuracy"] = np.mean(correct_mask)
        # print("Individual accuracy: {}".format(metrics['Accuracy']*100))

        prepositions = ['on', 'under', 'front', 'behind', 'left', 'right']
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        for i, d in enumerate(self.dataset):
            prep = list(set(prepositions).intersection(set(d['caption_options'][preds[i]].split())))
            gold_prep = list(set(prepositions).intersection(set(d['caption_options'][0].split())))
            #if len(prep) != 1 or len(gold_prep)!=1:
            #    pdb.set_trace()
            #    print("?")
            prep = prep[0]
            gold_prep = gold_prep[0]
            prep_counts[gold_prep][prep] += 1

            self.pred_dict[(d['image_path'].split('/')[-1].split('_')[0], \
                            d['image_path'].split('/')[-1].split('_')[-1][:-5])][d['image_path'].split('/')[-1].split('_')[1]] = prep
        #print(prep_counts)
        for d, correct in zip(self.dataset, correct_mask):
            self.eval_dict[(d['image_path'].split('/')[-1].split('_')[0], \
                            d['image_path'].split('/')[-1].split('_')[-1][:-5])][d['image_path'].split('/')[-1].split('_')[1]] = correct

        
        pair_correct = 0
        set_correct = 0
        for obj_pair, correct_dict in self.eval_dict.items():
            if correct_dict['left'] and correct_dict['right']:
                pair_correct += 1
            if self.subset == 'A':
                if correct_dict['on'] and correct_dict['under']:
                    pair_correct += 1
            else:
                if correct_dict['in-front'] and correct_dict['behind']:
                    pair_correct += 1
            if sum(correct_dict.values()) == 4:
                set_correct += 1
        pair_accuracy = pair_correct*100/(len(self.dataset)/2)
        set_accuracy = set_correct*100/(len(self.dataset)/4)
        # print("Pair accuracy: {}".format(pair_accuracy))
        # print("Set accuracy: {}".format(set_accuracy))
        all_prepositions = np.array(self.all_prepositions)

        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = (all_prepositions == prepositions)
            if prepositions_mask.sum() == 0:
                continue
            result_records.append({
                "Preposition": prepositions,
                "Accuracy": correct_mask[prepositions_mask].mean(),
                "Count": prepositions_mask.sum(),
                "Dataset": "Controlled Images - {}".format(self.subset)
            })
        individual_accuracy = metrics['Accuracy']
        return result_records, individual_accuracy


class COCO_QA(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir="", download=False, subset='one'):
        self.root_dir = root_dir
        if subset == 'one':
            annotation_file = os.path.join(root_dir, "coco_qa_one_obj.json")
            image_dir = os.path.join(root_dir, 'val2017')
        else:
            annotation_file = os.path.join(root_dir, "coco_qa_two_obj.json")
            image_dir = os.path.join(root_dir, 'val2017')


        self.dataset = json.load(open(annotation_file))
        self.subset = subset
        self.all_prepositions = []
        if self.subset == 'one':
            self.all_prepositions = [d[1].split()[-1] for d in self.dataset]
        else:
            for d in self.dataset:
                if 'left of' in d[1]:
                    self.all_prepositions.append('left')
                elif 'right of' in d[1]:
                    self.all_prepositions.append('right')
                elif 'above' in d[1]:
                    self.all_prepositions.append('above')
                else:
                    self.all_prepositions.append('below')
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(os.path.join(self.root_dir, 'val2017/{}.jpg'.format(str(test_case[0]).zfill(12)))).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        
        item = edict({"image_options": [image], "caption_options": [test_case[1], test_case[2]]})
        return item

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        metrics["Accuracy"] = np.mean(correct_mask)
        # print(metrics['Accuracy']*100)

        all_prepositions = np.array(self.all_prepositions)

        prepositions = list(set(self.all_prepositions))
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        opposite = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above', 'top': 'bottom', 'bottom': 'top'}

        for prep, pred in zip(self.all_prepositions, preds):
            if pred == 0:
                prep_counts[prep][prep] += 1
            else:
                prep_counts[prep][opposite[prep]] += 1
        #print(prep_counts)
        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = (all_prepositions == prepositions)
            if prepositions_mask.sum() == 0:
                continue
            result_records.append({
                "Preposition": prepositions,
                "Accuracy": correct_mask[prepositions_mask].mean(),
                "Count": prepositions_mask.sum(),
                "Dataset": "COCO-QA {}-object".format(self.subset)
            })
        individual_accuracy = metrics['Accuracy']
        return result_records, individual_accuracy

class VG_QA(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir="", download=False, subset='one'):
        self.root_dir = root_dir
        if subset == 'one':
            annotation_file = os.path.join(root_dir, "vg_qa_one_obj.json")
            image_dir = os.path.join(root_dir, 'vg_images')
        else:
            annotation_file = os.path.join(root_dir, "vg_qa_two_obj.json")
            image_dir = os.path.join(root_dir, 'vg_images')


        self.dataset = json.load(open(annotation_file))
        self.subset = subset
        self.all_prepositions = []
        if self.subset == 'one':
            self.all_prepositions = [d[1].split()[-1] for d in self.dataset]
        else:
            for d in self.dataset:
                if 'left of' in d[1]:
                    self.all_prepositions.append('left')
                elif 'right of' in d[1]:
                    self.all_prepositions.append('right')
                elif 'front of' in d[1]:
                    self.all_prepositions.append('front')
                elif 'behind' in d[1]:
                    self.all_prepositions.append('behind')
                else:
                    self.all_prepositions.append('top')
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(os.path.join(self.root_dir, 'vg_images/{}.jpg'.format(test_case[0]))).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        
        item = edict({"image_options": [image], "caption_options": [test_case[1], test_case[2]]})
        return item

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        metrics["Accuracy"] = np.mean(correct_mask)
        # print(metrics['Accuracy']*100)

        all_prepositions = np.array(self.all_prepositions)
        
        prepositions = list(set(self.all_prepositions)) + ['below', 'bottom', 'front']
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        opposite = {'left': 'right', 'right': 'left', 'behind': 'front', 'front': 'behind', 'above': 'below', 'below': 'above', 'bottom': 'top', 'top': 'bottom'}

        for prep, pred in zip(self.all_prepositions, preds):
            if pred == 0:
                prep_counts[prep][prep] += 1
            else:
                prep_counts[prep][opposite[prep]] += 1
        #print(prep_counts)
        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = (all_prepositions == prepositions)
            if prepositions_mask.sum() == 0:
                continue
            result_records.append({
                "Preposition": prepositions,
                "Accuracy": correct_mask[prepositions_mask].mean(),
                "Count": prepositions_mask.sum(),
                "Dataset": "VG-QA {}-object".format(self.subset)
            })
        individual_accuracy = metrics["Accuracy"]
        return result_records, individual_accuracy

if __name__ == '__main__':
    pretrained_path = "/your_ckpt_path/xxx.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_aro_relation(pretrained_path, device)
    evaluate_aro_attribute(pretrained_path, device)
    evaluate_sugar_crepe(pretrained_path, device)
    evaluate_winoground(pretrained_path, device)
    evaluate_valse(pretrained_path, device)
    evaluate_whats_up(pretrained_path, device)