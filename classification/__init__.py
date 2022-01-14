"""
这个项目主要提供了一个进行文本分类或聚类的接口。给定一个dataframe，标出其中的文本信息列以及标签列（如果有），就可以按照预先设定的算法进行
文本分类或者聚类。暂时支持tfidf，word2vec，bert等常用方法。
"""
from typing import List

import jieba
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
from transformers import BertTokenizer, BertModel


class QuickTextClassification:
    def __init__(self,
                 dataframe: pd.DataFrame,
                 text_columns: List = None,
                 label_column: int or str = None,
                 representation_algorithm: str = 'bow',
                 cluster_algorithm: str = None,
                 classification_algorithm: str = None,
                 bert_path: str = r"D:\pretrained_model\bert-base-chinese"
                 ):
        """supported algorithm: ['bow', 'tfidf', 'word2vec-sg','word2vec-cbow', 'bert']. if your desired algorithm is
        not implemented in QuickTextClassification, please use the features returned from """
        self.clusters = None
        self.cluster_model = None
        self.features = None
        self.text = None
        self.dataframe = dataframe
        if text_columns:
            if isinstance(text_columns[0], int):
                self.text_columns = self.dataframe.columns[text_columns]
            else:
                self.text_columns = text_columns
        else:
            # infer text columns from the dataframe
            self.dataframe = self.dataframe.convert_dtypes()
            self.text_columns = [col for col in self.dataframe.columns if self.dataframe[col].dtype == 'string']

        if isinstance(label_column, int):
            label_column = self.dataframe.columns[label_column]
        self.label_column = label_column
        self.representation_algorithm = representation_algorithm
        self.cluster_algorithm = cluster_algorithm
        self.classification_algorithm = classification_algorithm
        self.bert_path = bert_path

    @classmethod
    def from_filepath(cls, filename, *args, **kwargs):
        if filename.endswith('.xlsx'):
            dataframe = pd.read_excel(filename)
        elif filename.endswith('.csv'):
            dataframe = pd.read_csv(filename)
        return cls(dataframe, *args, **kwargs)

    def preprocess_text(self):
        from preprocessing import clean_text
        self.dataframe['text_all_in_one'] = clean_text(self.dataframe['text_all_in_one'])

    def concat_text_columns(self, return_numpy: bool = False):
        self.dataframe['text_all_in_one'] = self.dataframe[self.text_columns].fillna('').agg(' '.join, axis=1)
        self.preprocess_text()
        self.dataframe['text_all_in_one'] = self.dataframe['text_all_in_one'].apply(
            lambda x: ' '.join(jieba.cut(x)))
        self.text = self.dataframe['text_all_in_one'].to_numpy()
        if return_numpy:
            return self.text

    def extract_text_features(self):
        if self.text is None:
            self.concat_text_columns()
        if self.representation_algorithm in ['bow', 'tfidf']:
            if self.representation_algorithm == 'bow':
                from sklearn.feature_extraction.text import CountVectorizer
                model = CountVectorizer(ngram_range=(1, 3))
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                model = TfidfVectorizer(ngram_range=(1, 3))
            model.fit(self.text)
            return model.transform(self.text)

        elif self.representation_algorithm in ['word2vec-sg', 'word2vec-cbow']:
            pass

        elif self.representation_algorithm == 'bert':
            if self.bert_path:
                tokenizer = BertTokenizer.from_pretrained(self.bert_path)
                model = BertModel.from_pretrained(self.bert_path,
                                                  output_hidden_states=True)
            else:
                tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                model = BertModel.from_pretrained('bert-base-chinese',
                                                  output_hidden_states=True)
            model.eval()
            text_embeddings = []
            MAX_LENGTH = min(max([len(h) for h in self.text]), 512)
            NUM_EMBEDDING_LAYERS = 3
            with torch.no_grad():
                for sentence in self.text:
                    inputs = tokenizer.encode_plus(sentence,
                                                   padding='max_length',
                                                   max_length=MAX_LENGTH,
                                                   return_tensors='pt',
                                                   truncation=True)
                    outputs = model(**inputs)
                    hidden_states = outputs[2]  # change to 2 if using BERT model, change to 1 if using automodel
                    token_embeddings = torch.stack(hidden_states, dim=0)
                    token_embeddings = torch.squeeze(token_embeddings, dim=1)
                    token_embeddings = token_embeddings.permute(1, 0, 2)
                    sentence_embedding = []
                    for token in token_embeddings:
                        cat_vec = torch.flatten(token[-NUM_EMBEDDING_LAYERS:])
                        sentence_embedding.append(cat_vec)
                    sentence_embedding = torch.vstack(sentence_embedding).mean(dim=0)
                    text_embeddings.append(sentence_embedding)
                text_embeddings = torch.stack(text_embeddings, dim=0).numpy()

            return text_embeddings

        else:
            raise NotImplementedError

    def determine_optimal_num_clusters(self, min_num_cluster=2, max_num_cluster=10):
        # todo: 目前还是需要画图观察，然后确定一个最佳的 num_cluster, 有待改进
        num_clusters = range(min_num_cluster, max_num_cluster + 1)
        sse = {}
        for num_cluster in num_clusters:
            cluster_model = KMeans(n_clusters=num_cluster)
            features = self.extract_text_features()
            cluster_model.fit(X=features)
            sse[num_cluster] = cluster_model.inertia_
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()), '-o')
        plt.show()
        return

    def cluster(self, num_cluster, method=None):
        if method is None:
            self.features = self.extract_text_features()
        self.cluster_model = KMeans(n_clusters=num_cluster)
        self.cluster_model.fit_transform(self.features)
        self.clusters = self.cluster_model.labels_


if __name__ == "__main__":
    pass







