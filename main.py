import time
from pathlib import Path
from src.sella import Sella
import argparse

def ensure_dirs(args):
    if not args.multi:
        (args.data_home / args.dataset / "exp_log").mkdir(parents=True, exist_ok=True)
        args.output_dir = args.data_home / args.dataset / "output" / args.version
        args.log_dir = args.data_home / args.dataset / "exp_log" / args.version
        args.lf_dir = args.data_home / args.dataset / "label_function" / args.version
        args.log_dir.mkdir(parents=True, exist_ok=True)
    else:
        (args.data_home / args.dataset / args.label_num / "exp_log").mkdir(parents=True, exist_ok=True)
        args.output_dir = args.data_home / args.dataset / args.label_num / "output" / args.version
        args.log_dir = args.data_home / args.dataset / args.label_num / "exp_log" / args.version
        args.lf_dir = args.data_home / args.dataset / args.label_num / "label_function" / args.version
        args.log_dir.mkdir(parents=True, exist_ok=True)

def main(args):
    ensure_dirs(args)
    sella = Sella(args)
    sella.run_exp()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sella experiments")

    # === general ===
    parser.add_argument("--cur_time", type=float, default=time.time(), help="Current timestamp")
    parser.add_argument("--data_home", type=Path, default=Path("data"), help="Root data directory")
    parser.add_argument("--dataset", type=str, default="youtube", help="Dataset name")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--dev_size", type=float, default=0.5)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # === model / llm config ===
    parser.add_argument("--llm_model", type=str, default="google/gemini-2.5-flash", help="Model used for LF generation")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_token", type=int, default=10000)

    # === LF generation ===
    parser.add_argument("--min_lf_per_type", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--labeled_percentage", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_patience", type=int, default=5)
    parser.add_argument("--eval_times", type=int, default=5)
    parser.add_argument("--run_saved_lfs", action="store_true", help="Run with saved label functions")

    # === experiment config ===
    parser.add_argument("--multi", action="store_true", help="Multi-label mode")
    parser.add_argument("--label_num", type=str, default="", help="Specific label name in multi-label setting")
    parser.add_argument("--eval_metric", type=str, default="f1_weighted")

    # === models ===
    parser.add_argument("--labeling_models", nargs="+", default=["Snorkel"], help="Labeling model list")
    parser.add_argument("--end_models", nargs="+", default=["MLP"], help="End model list")

    # === API ===
    parser.add_argument("--api_key", type=str, default="", help="API key for provider if needed")
    
    
    args = parser.parse_args()
    
    main(args)