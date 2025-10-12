import time
import json
import numpy as np

from pathlib import Path

import torch
from src.utils import get_labeled_subset, get_wrench_model, generate_semantic_lfs, generate_structural_lfs, generate_surface_lfs, print_label_distribution, print_label_function_stats
from wrench.dataset import load_dataset
from .label_function.base import LFType
from .label_function.surface import SurfaceLF
from .label_function.structural import StructuralLF
from .label_function.semantic import SemanticLF
from sklearn.metrics import accuracy_score, f1_score

class Sella:
    def __init__(self, args):
        self.args = args
        self.cur_surface_lf = []
        self.cur_structural_lf = []
        self.cur_semantic_lf = []
        self.patience = 0
        self.results = {}
        self.timing = {}
        
        self.train_data, self.valid_data, self.test_data = load_dataset(
            data_home=self.args.data_home,
            dataset=self.args.dataset,
            extract_feature=True,
            extract_fn='bert',
            model_name='bert-base-cased',
            cache_name='bert'
        )
                
        self.valid_data = get_labeled_subset(
            dataset=self.valid_data, 
            labeled_percentage=self.args.labeled_percentage
        )
        
        print_label_distribution("Train", self.train_data)
        print_label_distribution("Valid", self.valid_data)
        print_label_distribution("Test", self.test_data)
        self.LM_NAMES = self.args.labeling_models
        self.EM_NAMES = self.args.end_models
        
    def calc_acceptable_threshold(self, lfs, is_intra=True):
        if not lfs or len(lfs) == 0:
            return 0.0, 0.0
        max_perf = np.array([lf.estimated_performance for lf in lfs]).max()
        # max_cov = np.array([lf.estimated_coverage for lf in lfs]).max()
        if is_intra:
            return max_perf * self.args.alpha, 0
        else:
            return max_perf * self.args.alpha / 2, 0

    def filter_lf(self, lf, acceptable_performance, acceptable_coverage):
        filtered_lf = []
        for i, h in enumerate(lf):
            if h.estimated_performance < acceptable_performance:
                print(f"Skipping lf with estimated_performance {h.estimated_performance:.4f} below min acceptable_performance {acceptable_performance:.4f}\n\n")
            elif  h.estimated_coverage < acceptable_coverage:
                print(f"Skipping lf with estimated_coverage {h.estimated_coverage:.4f} below min acceptable_coverage {acceptable_coverage:.4f}\n\n")
            else:
                filtered_lf.append(h)
        return filtered_lf

    def load_lf_from_file(self, lf_type):
        lfs = []
        if lf_type == LFType.SURFACE:
            lf_files = list(self.args.lf_dir.glob("*.py"))
            for file in lf_files:
                lf = SurfaceLF(saved_path=file, lf_dir=self.args.lf_dir)
                lf.estimate_performance(self.valid_data, self.train_data, beta=self.args.beta)
                print_label_function_stats(lf)
                lfs.append(lf)
        else:
            lf_dir = self.args.lf_dir / lf_type.value
            lf_files = list(lf_dir.glob("*.pkl"))
            for file in lf_files:
                if lf_type == LFType.STRUCTURAL:
                    lf = StructuralLF(saved_path=file, lf_dir=self.args.lf_dir)
                elif lf_type == LFType.SEMANTIC:
                    lf = SemanticLF(saved_path=file, lf_dir=self.args.lf_dir)
                else:
                    raise ValueError(f"Unknown LF type: {lf_type}")
                
                lf.find_threshold(self.valid_data, self.train_data, beta=self.args.beta)
                print_label_function_stats(lf)
                lfs.append(lf)
        return lfs

    def generate_lf(self, lf_type, quantity):
        if lf_type == LFType.SURFACE:
            if self.args.run_saved_lfs and self.patience == 1:
                lfs = self.load_lf_from_file(lf_type)
            else:
                lfs = generate_surface_lfs(self.valid_data, self.train_data, quantity, self.args)
        elif lf_type == LFType.STRUCTURAL:
            if self.args.run_saved_lfs and self.patience == 1:
                lfs = self.load_lf_from_file(lf_type)
            else:
                lfs = generate_structural_lfs(self.valid_data, self.train_data, quantity, self.args)
        elif lf_type == LFType.SEMANTIC:
            if self.args.run_saved_lfs and self.patience == 1:
                lfs = self.load_lf_from_file(lf_type)
            else:
                lfs = generate_semantic_lfs(self.valid_data, self.train_data, quantity, self.args)
        else:
            lfs = []
        return lfs
    
    def filter_intra(self, lf_type):
        if lf_type == LFType.SURFACE:
            acp_prec, acp_cov = self.calc_acceptable_threshold(self.cur_surface_lf, is_intra=True)
            self.cur_surface_lf = self.filter_lf(self.cur_surface_lf, acp_prec, acp_cov)
        elif lf_type == LFType.STRUCTURAL:
            acp_prec, acp_cov = self.calc_acceptable_threshold(self.cur_structural_lf, is_intra=True)
            self.cur_structural_lf = self.filter_lf(self.cur_structural_lf, acp_prec, acp_cov)
        elif lf_type == LFType.SEMANTIC:
            acp_prec, acp_cov = self.calc_acceptable_threshold(self.cur_semantic_lf, is_intra=True)
            self.cur_semantic_lf = self.filter_lf(self.cur_semantic_lf, acp_prec, acp_cov)

    def filter_inter(self):
        lfs =  self.cur_surface_lf + self.cur_structural_lf + self.cur_semantic_lf
        acp_prec, acp_cov = self.calc_acceptable_threshold(lfs, is_intra=False)
        
        self.cur_surface_lf = self.filter_lf(self.cur_surface_lf, acp_prec, acp_cov)
        self.cur_structural_lf = self.filter_lf(self.cur_structural_lf, acp_prec, acp_cov)
        self.cur_semantic_lf = self.filter_lf(self.cur_semantic_lf, acp_prec, acp_cov)

    def check_termination(self):
        if self.patience > self.args.max_patience:
            return False
        if len(self.cur_surface_lf) < self.args.min_lf_per_type:
            self.patience += 1
            return True
        if len(self.cur_structural_lf) < self.args.min_lf_per_type:
            self.patience += 1
            return True
        if len(self.cur_semantic_lf) < self.args.min_lf_per_type:
            self.patience += 1
            return True
        return False

    def update_weak_labels(self):
        selected_lf = self.cur_surface_lf + self.cur_structural_lf + self.cur_semantic_lf

        self.train_data.weak_labels = np.array([h.get_labels(self.train_data) for h in selected_lf]).T
        self.valid_data.weak_labels = np.array([h.get_labels(self.valid_data) for h in selected_lf]).T
        self.test_data.weak_labels = np.array([h.get_labels(self.test_data) for h in selected_lf]).T

    def aggregate_weak_labels(self):
        selected_lf = self.cur_surface_lf + self.cur_structural_lf + self.cur_semantic_lf

        self.train_data.weak_labels = np.array([h.get_labels(self.train_data) for h in selected_lf]).T
        self.valid_data.weak_labels = np.array([h.get_labels(self.valid_data) for h in selected_lf]).T
        self.test_data.weak_labels = np.array([h.get_labels(self.test_data) for h in selected_lf]).T
        
        label_model = get_wrench_model(self.args.labeling_models[0])
        label_model.fit(
            dataset_train=self.train_data, 
            dataset_valid=self.valid_data,
            verbose=True,
        )
            
        lm_acc = label_model.test(self.valid_data, 'acc')
        lm_f1 = label_model.test(self.valid_data, 'f1_weighted')

        print(f"Label Model {self.args.labeling_models[0]} - Valid Acc: {lm_acc:.4f}, Valid F1: {lm_f1:.4f}")
        return label_model.predict(self.train_data)
        

    def evaluate_lfs(self):
        self.results = {}
        train_data_covered = self.train_data.get_covered_subset()
        coverage = len(train_data_covered) / len(self.train_data)
        
        for label_model_name in self.LM_NAMES:
            lm_acc_array = np.zeros(self.args.eval_times)
            lm_f1_array = np.zeros(self.args.eval_times)
            lm_collection = []
        
            for T1 in range(self.args.eval_times):
                label_model = get_wrench_model(label_model_name)
                label_model.fit(
                    dataset_train=self.train_data, 
                    dataset_valid=self.valid_data,
                    verbose=False,
                )
                
                lm_acc = label_model.test(self.test_data, 'acc')
                lm_f1 = label_model.test(self.test_data, 'f1_weighted')
        
                lm_acc_array[T1] = lm_acc
                lm_f1_array[T1] = lm_f1
                lm_collection.append(label_model)
        
            lm_acc_mean, lm_acc_std = np.mean(lm_acc_array), np.std(lm_acc_array)
            lm_f1_mean, lm_f1_std = np.mean(lm_f1_array), np.std(lm_f1_array)
            self.results.update({
                f"{label_model_name}_acc_mean": round(lm_acc_mean, 5),
                f"{label_model_name}_acc_std": round(lm_acc_std, 5),
                f"{label_model_name}_f1_mean": round(lm_f1_mean, 5),
                f"{label_model_name}_f1_std": round(lm_f1_std, 5),
            })
        
            best_label_model_index = np.argmax(lm_acc_array)
            best_label_model = lm_collection[best_label_model_index]
            
            train_prob_label_covered = best_label_model.predict_proba(train_data_covered)
            train_label_covered = best_label_model.predict(train_data_covered)
            
            lm_acc_train = accuracy_score(train_data_covered.labels, train_label_covered)
            lm_f1_train = f1_score(train_data_covered.labels, train_label_covered, average="weighted")
            
            self.results.update({
                "covered_train_acc": lm_acc_train,
                "covered_train_f1": lm_f1_train,
                "train_coverage": round(coverage, 5),
                "train_covered_size": len(train_data_covered.examples),
            })
            
            print(f"Best Label Model {label_model_name} - Covered Train Acc: {lm_acc_train:.4f}, Covered Train F1: {lm_f1_train:.4f}, Coverage: {coverage:.4f}")
        
            for end_model_name in self.EM_NAMES:
                em_acc_array = np.zeros(self.args.eval_times)
                em_f1_array = np.zeros(self.args.eval_times)
        
                for T2 in range(self.args.eval_times):
                    end_model = get_wrench_model(end_model_name)
                    
                    end_model.fit(
                        dataset_train=train_data_covered,
                        y_train=train_prob_label_covered,
                        dataset_valid=self.valid_data,
                        evaluation_step=100,
                        metric=self.args.eval_metric,
                        verbose=False,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                    
                    em_acc = end_model.test(self.test_data, 'acc')
                    em_f1 = end_model.test(self.test_data, 'f1_weighted')
                    
                    em_acc_array[T2] = em_acc
                    em_f1_array[T2] = em_f1
                    
                    print(f"{T2}. Label Model {label_model_name} + End Model {end_model_name} - Test Acc: {em_acc:.4f}, Test F1: {em_f1:.4f}")
        
                em_acc_mean, em_acc_std = np.mean(em_acc_array), np.std(em_acc_array)
                em_f1_mean, em_f1_std = np.mean(em_f1_array), np.std(em_f1_array)
                results = {
                    f"{label_model_name}_{end_model_name}_acc_mean": round(em_acc_mean, 5),
                    f"{label_model_name}_{end_model_name}_acc_std": round(em_acc_std, 5),
                    f"{label_model_name}_{end_model_name}_f1_mean": round(em_f1_mean, 5),
                    f"{label_model_name}_{end_model_name}_f1_std": round(em_f1_std, 5),
                }
                print(f"Label Model {label_model_name} + End Model {end_model_name} - Test Acc Mean: {em_acc_mean:.4f}, Test Acc Std: {em_acc_std:.4f}, Test F1 Mean: {em_f1_mean:.4f}, Test F1 Std: {em_f1_std:.4f}")
                self.results.update(results)
    
    def save_results_json(self, output_dir="output"):
        if not hasattr(self, "results") or not self.results:
            raise ValueError("No results to save. Please run evaluate_lfs() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{self.args.version}_{self.args.cur_time}.json"
        
        data_to_save = {
            "dataset": self.args.dataset,
            "version": self.args.version,
            "time": self.args.cur_time,
            "results": self.results,
            "timing": self.timing
        }
        
        with open(output_file, 'w') as f:
            json.dump(data_to_save, f, indent=4, sort_keys=True)
                
    def display_stats(self):
        for split, data in zip(["train", "valid", "test"], [self.train_data, self.valid_data, self.test_data]):
            print(f"{split.capitalize()} Data LF summary:")
            print(data.lf_summary())
    
    def run_exp(self):
        total_start = time.time()
        print("Starting Sella Experiment...")
    
        self.timing = {
            "generate_lf": 0.0,
            "filter_intra": 0.0,
            "filter_inter": 0.0,
            "update_weak_labels": 0.0,
            "evaluate_lfs": 0.0
        }
    
        while self.check_termination():
            gen_start = time.time()
            if self.args.min_lf_per_type - len(self.cur_surface_lf) > 0:
                print(f"Generating {self.args.min_lf_per_type - len(self.cur_surface_lf)} surface LFs...")
                self.cur_surface_lf.extend(
                    self.generate_lf(LFType.SURFACE, self.args.min_lf_per_type - len(self.cur_surface_lf))
                )
            if self.args.min_lf_per_type - len(self.cur_structural_lf) > 0:
                print(f"Generating {self.args.min_lf_per_type - len(self.cur_structural_lf)} structural LFs...")
                self.cur_structural_lf.extend(
                    self.generate_lf(LFType.STRUCTURAL, self.args.min_lf_per_type - len(self.cur_structural_lf))
                )
            if self.args.min_lf_per_type - len(self.cur_semantic_lf) > 0:
                print(f"Generating {self.args.min_lf_per_type - len(self.cur_semantic_lf)} semantic LFs...")
                self.cur_semantic_lf.extend(
                    self.generate_lf(LFType.SEMANTIC, self.args.min_lf_per_type - len(self.cur_semantic_lf))
                )
            self.timing["generate_lf"] += time.time() - gen_start
    
            if self.args.alpha == 0:
                break
    
            intra_start = time.time()
            self.filter_intra(LFType.SURFACE)
            self.filter_intra(LFType.STRUCTURAL)
            self.filter_intra(LFType.SEMANTIC)
            self.timing["filter_intra"] += time.time() - intra_start
    
        inter_start = time.time()
        self.filter_inter()
        self.timing["filter_inter"] += time.time() - inter_start
    
        start = time.time()
        self.update_weak_labels()
        self.timing["update_weak_labels"] += time.time() - start
    
        self.display_stats()
    
        start = time.time()
        self.evaluate_lfs()
        self.timing["evaluate_lfs"] += time.time() - start
        
        self.timing["total"] = time.time() - total_start
        self.timing = {k: round(v, 4) for k, v in self.timing.items()}
        
        if not self.args.run_saved_lfs:
            for lf in self.cur_structural_lf:
                lf.save()
            for lf in self.cur_semantic_lf:
                lf.save()
        self.save_results_json()
        
    def run(self):
        while self.check_termination():
            self.cur_surface_lf.extend(
                self.generate_lf(LFType.SURFACE, self.args.min_lf_per_type - len(self.cur_surface_lf))
            )
            self.cur_structural_lf.extend(
                self.generate_lf(LFType.STRUCTURAL, self.args.min_lf_per_type - len(self.cur_structural_lf))
            )
            self.cur_semantic_lf.extend(
                self.generate_lf(LFType.SEMANTIC, self.args.min_lf_per_type - len(self.cur_semantic_lf))
            )
    
            if self.args.alpha == 0:
                break
    
            self.filter_intra(LFType.SURFACE)
            self.filter_intra(LFType.STRUCTURAL)
            self.filter_intra(LFType.SEMANTIC)

            self.filter_inter()
            
        self.update_weak_labels()
        return self.aggregate_weak_labels()