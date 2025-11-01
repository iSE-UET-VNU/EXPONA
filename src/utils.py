import argparse
from typing import List
import random
import pandas as pd
import os.path as osp
from sklearn.model_selection import train_test_split
import json
import numpy as np

from wrench.endmodel import EndClassifierModel, LogRegModel
from wrench.labelmodel import Snorkel, MajorityVoting, MajorityWeightedVoting, FlyingSquid, DawidSkene
from wrench.dataset.basedataset import BaseDataset

from .generator import LLMGenerator, ScikitGenerator
from .label_function.surface import SurfaceLF
from .label_function.structural import StructuralLF
from .label_function.semantic import SemanticLF
from .prompts.utils import create_system_prompt, create_user_prompt, extract_response

from collections import Counter

def df2json(df):
    return {str(i): {
                "data": {"text": row["text"]},
                "label": row["label"],
                "weak_labels": []
            }
            for i, row in df.iterrows()
            }

def print_label_distribution(name, dataset):
    labels = np.array(dataset.labels)
    counter = Counter(labels)
    total = len(labels)
    print(f"\n{name} Label Distribution:")
    for label, count in sorted(counter.items()):
        print(f"  Label {label}: {count} ({count/total:.2%})")

    print(f"Total: {total}")

def convert_wrench_dataset(self):
    df = pd.read_csv(osp.join(self.args.dataset_path, self.args.dataset + ".csv"))
    
    train_df, temp_df = train_test_split(df, train_size=self.args.train_size, random_state=42)
    valid_df, test_df = train_test_split(temp_df, train_size=self.args.val_size, random_state=42)
    
    self.train_df = train_df
    self.valid_df = valid_df
    self.test_df = test_df

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_json = df2json(train_df)
    valid_json = df2json(valid_df)
    test_json = df2json(test_df)

    with open(f"{self.args.dataset_path}/{self.args.dataset}/train.json", "w", encoding="utf-8") as f:
        json.dump(train_json, f, ensure_ascii=False, indent=2)

    with open(f"{self.args.dataset_path}/{self.args.dataset}/valid.json", "w", encoding="utf-8") as f:
        json.dump(valid_json, f, ensure_ascii=False, indent=2)

    with open(f"{self.args.dataset_path}/{self.args.dataset}/test.json", "w", encoding="utf-8") as f:
        json.dump(test_json, f, ensure_ascii=False, indent=2)
        
    return True


def get_wrench_model(name: str):
    if name == "FS":
        return FlyingSquid()
    elif name == "DS":
        return DawidSkene()
    if name == "WMV":
        return MajorityWeightedVoting()
    elif name == "MV":
        return MajorityVoting()
    elif name == "Snorkel":
        return Snorkel()
    elif name == "MLP":
        return EndClassifierModel(backbone="MLP")
    elif name == "LR":
        return LogRegModel()
    else:
        raise ValueError(f"Unknown model name: {name}")
    
def get_labeled_subset(dataset, labeled_percentage: float = 0.15, min_num_labeled: int = 100, min_per_class: int = 5):
    num_dataset = len(dataset)
    num_subset = int(labeled_percentage * num_dataset)

    labels = np.array(dataset.labels)
    unique_classes = np.unique(labels)

    selected_indices = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        chosen = np.random.choice(cls_indices, size=min(min_per_class, len(cls_indices)), replace=False)
        selected_indices.extend(chosen)

    remaining_indices = list(set(range(num_dataset)) - set(selected_indices))
    if len(selected_indices) < num_subset:
        extra_needed = num_subset - len(selected_indices)
        extra_chosen = np.random.choice(remaining_indices, size=extra_needed, replace=False)
        selected_indices.extend(extra_chosen)

    return dataset.create_subset(selected_indices)

def print_label_function_stats(heuristic):
        print("="*50)
        print(f"Type of LF: {heuristic.__class__}")
        print(f"Estimated Performance (labeled): {heuristic.estimated_performance:.4f}")
        print(f"Estimated Coverage (unlabeled): {heuristic.estimated_coverage:.4f}")
        print(f"Estimated F_beta: {heuristic.estimated_fbeta:.4f}")
        print("="*50)

def generate_surface_lfs(
    labeled_dataset: BaseDataset,
    unlabeled_dataset: BaseDataset,
    num_trials: int,
    args: argparse.Namespace,
) -> List:
    
    if args.multi:
        system_prompt = create_system_prompt()
        user_prompt = create_user_prompt(dataset_name=args.dataset, label_number=int(args.label_num))
    else:
        system_prompt = create_system_prompt()
        user_prompt = create_user_prompt(dataset_name=args.dataset)
    agent = LLMGenerator(system_prompt=system_prompt, args=args)
    
    lf_list = []
    response_list = agent.get_completion(user_prompt, n=num_trials)
    for response in response_list:
        code_str = extract_response(response.content)
        if SurfaceLF.is_runnable_lf(code_str=code_str, test_input=labeled_dataset.examples[0]["text"]):
            lf = SurfaceLF(raw_code=code_str, eval_metric=args.eval_metric, lf_path=args.lf_dir)
            lf.estimate_performance(labeled_dataset=labeled_dataset, 
                                           unlabeled_dataset=unlabeled_dataset,
                                           beta=args.beta)
            lf_list.append(lf)
            
            print_label_function_stats(lf)

    return lf_list   

def generate_structural_lfs(
    labeled_dataset: BaseDataset,
    unlabeled_dataset: BaseDataset,
    num_trials: int,
    args: argparse.Namespace,
) -> List:

    lf_list = []

    for _ in range(num_trials):
        train_idx = range(len(labeled_dataset))
    
        drop_ratio = random.uniform(0.05, 0.15)
        n_drop = int(len(train_idx) * drop_ratio)
        
        if n_drop > 0:
            drop_idx = np.random.choice(train_idx, size=n_drop, replace=False)
            train_idx = list(set(train_idx) - set(drop_idx))
    
        train_labeled = labeled_dataset.create_subset(train_idx)
    
        
        generator = ScikitGenerator(train_labeled.features_2, train_labeled.labels)
        lf = generator.generate_lfs(model='svm')
        lf = StructuralLF(lf=lf,
                        eval_metric=args.eval_metric,
                        lf_path=args.lf_dir,
                        )
        lf.find_threshold(labeled_dataset=labeled_dataset, 
                            unlabeled_dataset=unlabeled_dataset, 
                            beta=args.beta)
        lf_list.append(lf)

        print_label_function_stats(lf)
    
    return lf_list

def generate_semantic_lfs(
    labeled_dataset: BaseDataset,
    unlabeled_dataset: BaseDataset,
    num_trials: int,
    args: argparse.Namespace,
) -> List:

    lf_list = []

    for _ in range(num_trials):
        train_idx = range(len(labeled_dataset))
        drop_ratio = random.uniform(0.05, 0.2)
        n_drop = int(len(train_idx) * drop_ratio)
        if n_drop > 0:
            drop_idx = np.random.choice(train_idx, size=n_drop, replace=False)
            train_idx = list(set(train_idx) - set(drop_idx))
    
        train_labeled = labeled_dataset.create_subset(train_idx)
        
        
        generator = ScikitGenerator(train_labeled.features, train_labeled.labels)
        lf = generator.generate_lfs(model='mlp')

        lf = SemanticLF(lf=lf, 
                        eval_metric=args.eval_metric,
                        lf_path=args.lf_dir,
                        )
        lf.find_threshold(labeled_dataset=labeled_dataset, 
                                    unlabeled_dataset=unlabeled_dataset, 
                                    beta=args.beta)
        lf_list.append(lf)

        print_label_function_stats(lf)
    
    return lf_list