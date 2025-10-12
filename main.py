from src.sella import Sella
from src.config import Config

def ensure_dirs(args):
    (args.data_home / args.dataset / "exp_log").mkdir(parents=True, exist_ok=True)
    args.log_dir = args.data_home / args.dataset / "exp_log" / args.version
    args.lf_dir = args.data_home / args.dataset / "label_function" / args.version
    args.log_dir.mkdir(parents=True, exist_ok=True)
    return args

def main():
    args = Config()
    args = ensure_dirs(args)
    sella = Sella(args)
    sella.run_exp()
    
if __name__ == "__main__":
    main()