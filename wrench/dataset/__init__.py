from .basedataset import BaseDataset
from .dataset import NumericDataset, TextDataset, RelationDataset, ImageDataset
from .seqdataset import BaseSeqDataset
from .torchdataset import sample_batch, TorchDataset, BERTTorchTextClassDataset, BERTTorchRelationClassDataset, ImageTorchDataset

numeric_datasets = ['census', 'mushroom', 'spambase', 'PhishingWebsites', 'Bioresponse', 'bank-marketing', 'basketball', 'tennis', 'commercial']
text_datasets = ['agnews', 'imdb', 'sms', 'trec', 'yelp', 'youtube', 'medabs', 'clickbait', 'finance', 'tos', 'chemprot', 'massive', 'paper', 'pubmed', 'movie_genre']
relation_dataset = ['cdr', 'spouse', 'chemprot', 'semeval']
cls_dataset_list = numeric_datasets + text_datasets + relation_dataset
bin_cls_dataset_list = numeric_datasets + ['cdr', 'spouse', 'sms', 'yelp', 'imdb', 'youtube']
seq_dataset_list = ['laptopreview', 'ontonotes', 'ncbi-disease', 'bc5cdr', 'mit-restaurants', 'mit-movies', 'wikigold', 'conll']

import shutil
from pathlib import Path
from os import environ, makedirs
from os.path import expanduser, join


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
import numpy as np
import logging

class EmbeddingFactory:
    def __init__(self, method: str = 'tfidf', max_features: int = 5000):
        self.method = method
        self.vectorizer = None
        self.stat_scores = None
        self.max_features = max_features
        self.fitted = False

    def fit(self, texts, labels=None):
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                stop_words='english'
            )
            self.vectorizer.fit(texts)
            self.fitted = True

        elif self.method in ['chi2', 'mi', 'bow']:
            self.vectorizer = CountVectorizer(max_features=self.max_features)
            X = self.vectorizer.fit_transform(texts)

            if self.method == 'chi2':
                if labels is None:
                    raise ValueError("Labels are required for chi2")
                self.stat_scores, _ = chi2(X, labels)

            elif self.method == 'mi':
                if labels is None:
                    raise ValueError("Labels are required for mutual information")
                self.stat_scores = mutual_info_classif(X, labels)

            elif self.method == 'bow':
                self.stat_scores = None  # no weighting

            self.fitted = True

        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def transform(self, texts):
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()")

        X = self.vectorizer.transform(texts).toarray()

        if self.stat_scores is not None:
            return X * self.stat_scores  # element-wise scaling
        return X


#### dataset downloading and loading
def get_data_home(data_home=None) -> str:
    data_home = data_home or environ.get('WRENCH_DATA', join('~', 'wrench_data'))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def get_dataset_type(dataset_name):
    if dataset_name in numeric_datasets:
        return NumericDataset
    elif dataset_name in text_datasets:
        return TextDataset
    elif dataset_name in relation_dataset:
        return RelationDataset
    elif dataset_name in seq_dataset_list:
        return BaseSeqDataset
    raise NotImplementedError('cannot recognize the dataset type! please specify the dataset_type.')

def extract_texts(examples):
    if isinstance(examples[0], dict) and 'text' in examples[0]:
        return [e['text'] for e in examples]
    return examples

def load_dataset(data_home, dataset, dataset_type=None, label_number=None, extract_feature=False, extract_fn=None, **kwargs):
    if dataset_type is None:
        dataset_class = get_dataset_type(dataset)
    else:
        dataset_class = eval(dataset_type)
    
    if label_number:
        dataset_path = Path(data_home) / dataset / label_number
    else:
        dataset_path = Path(data_home) / dataset
    train_data = dataset_class(path=dataset_path, split='train')
    valid_data = dataset_class(path=dataset_path, split='valid')
    test_data = dataset_class(path=dataset_path, split='test')

    if extract_feature and (dataset_class != BaseSeqDataset):
        extractor_fn = train_data.extract_feature(extract_fn=extract_fn, return_extractor=True, **kwargs)
        valid_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)
        test_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)
        
    factory = EmbeddingFactory()
    factory.fit(extract_texts(train_data.examples))

    train_data.features_2 = factory.transform(extract_texts(train_data.examples))
    valid_data.features_2 = factory.transform(extract_texts(valid_data.examples))
    test_data.features_2 = factory.transform(extract_texts(test_data.examples))

    return train_data, valid_data, test_data


def load_image_dataset(data_home, dataset, image_root_path, preload_image=True, extract_feature=False, extract_fn='pretrain', **kwargs):
    dataset_path = Path(data_home) / dataset
    train_data = ImageDataset(path=dataset_path, split='train', image_root_path=image_root_path, preload_image=preload_image)
    valid_data = ImageDataset(path=dataset_path, split='valid', image_root_path=image_root_path, preload_image=preload_image)
    test_data = ImageDataset(path=dataset_path, split='test', image_root_path=image_root_path, preload_image=preload_image)

    if extract_feature:
        extractor_fn = train_data.extract_feature(extract_fn=extract_fn, return_extractor=True, **kwargs)
        valid_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)
        test_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)

    return train_data, valid_data, test_data
