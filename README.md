<div align="center">

# EXPONA: Structured Exploration and Exploitation of Label Functions for Automated Data Annotation
</div>


## Introduction

## The Architecture

## Quick Start
### Prerequisites
```bash
python3 -m venv expona
bash expona/bin/activate
pip install -r requirements.txt
```
### Datasets
All datasets should be stored inside the `./data` directory. For convenience, you can directly download public datasets via [Kaggle]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/phonglmnguynduy/expona-datasets/data)). 

If you want to use **your own dataset**, register it in `./wrench/dataset/__init__.py` by adding its name and comply WRENCH format as above examples.

### Running

#### Ag News
```bash
python3 main.py --dataset ag_news --min-lf-per-type 20 --max-patience 5 --alpha 0.9 --beta 0.1
```

#### IMDB
```bash
python3 main.py --dataset imdb --min-lf-per-type 20 --max-patience 5 --alpha 0.9 --beta 0.1
```

## Contact us
If you have any questions, comments, or suggestions, please do not hesitate to contact us.
- Email: 22028164@vnu.edu.vn
