import sys
import time
from pathlib import Path
from src.expona import Expona
import argparse
from dotenv import load_dotenv
from loguru import logger


def setup_logging(verbose: bool, log_file=None):
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    if log_file is not None:
        logger.add(
            log_file,
            level=level,
            format="{time:HH:mm:ss} | {level: <8} | {message}",
        )

def ensure_dirs(args):
    if not args.multi:
        (args.data_home / args.dataset / "exp_log").mkdir(parents=True, exist_ok=True)
        args.output_dir = args.data_home / args.dataset / "output" / args.exp_name
        args.log_dir = args.data_home / args.dataset / "exp_log" / args.exp_name
        args.lf_dir = args.data_home / args.dataset / "label_function" / args.exp_name
        args.log_dir.mkdir(parents=True, exist_ok=True)
    else:
        (args.data_home / args.dataset / args.label_num / "exp_log").mkdir(parents=True, exist_ok=True)
        args.output_dir = args.data_home / args.dataset / args.label_num / "output" / args.exp_name
        args.log_dir = args.data_home / args.dataset / args.label_num / "exp_log" / args.exp_name
        args.lf_dir = args.data_home / args.dataset / args.label_num / "label_function" / args.exp_name
        args.log_dir.mkdir(parents=True, exist_ok=True)

def main(args):
    load_dotenv()
    ensure_dirs(args)
    setup_logging(args.verbose, args.log_dir / f"experiment_{args.cur_time}.log")
    expona = Expona(args)
    expona.run_exp()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run expona experiments")

    # === general ===
    parser.add_argument("--data_home", type=Path, default=Path("data"), help="Root data directory")
    parser.add_argument("--dataset", type=str, default="massive", help="Dataset name")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--dev_size", type=float, default=0.5)
    parser.add_argument("--exp_name", type=str, default="v1")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # === LF generation ===
    parser.add_argument("--surface_lf_model", type=str, default="openai/gpt-4.1", help="Model used for surface LF generation")
    parser.add_argument("--structural_lf_model", type=str, default="lr", help="Model used for structural LF generation")
    parser.add_argument("--semantic_lf_model", type=str, default="mlp", help="Model used for semantic LF generation")
    parser.add_argument("--min_lf_per_type", type=int, default=20, help="Minimum number of LFs to generate for each type")
    parser.add_argument("--alpha", type=float, default=0.9, help="Alpha parameter for LF selection")
    parser.add_argument("--labeled_percentage", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for threshold finding")
    parser.add_argument("--max_patience", type=int, default=5)
    parser.add_argument("--eval_times", type=int, default=5)
    parser.add_argument("--run_saved_lfs", action="store_true", help="Run with saved label functions")
    parser.add_argument("--using_unlabeled", action="store_true", help="Whether to use unlabeled data in threshold finding")

    # === experiment config ===
    parser.add_argument("--multi", action="store_true", help="Multi-label mode")
    parser.add_argument("--label_num", type=str, default="", help="Specific label name in multi-label setting")
    parser.add_argument("--eval_metric", type=str, default="f1_weighted")

    # === models ===
    parser.add_argument("--labeling_models", nargs="+", default=["Snorkel"], help="Labeling model list")
    parser.add_argument("--end_models", nargs="+", default=["MLP"], help="End model list")
    
    args = parser.parse_args()
    args.cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    main(args)