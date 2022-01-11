"""
这个项目主要提供了一个进行文本分类或聚类的接口。给定一个dataframe，标出其中的文本信息列以及标签列（如果有），就可以按照预先设定的算法进行
文本分类或者聚类。暂时支持tfidf，word2vec，bert等常用方法。
"""
from typing import List

import jieba
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class QuickTextClassification:
    def __init__(self,
                 dataframe: pd.DataFrame,
                 text_columns: List = None,
                 label_column: int or str = None,
                 representation_algorithm: str = 'bow',
                 cluster_algorithm: str = None,
                 classification_algorithm: str = None
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
        if self.representation_algorithm in ['bow', 'tfidf']:
            if self.representation_algorithm == 'bow':
                from sklearn.feature_extraction.text import CountVectorizer
                model = CountVectorizer(ngram_range=(1, 3))
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                model = TfidfVectorizer(ngram_range=(1, 3))
            if self.text is None:
                self.concat_text_columns()
            model.fit(self.text)
            return model.transform(self.text)

        elif self.representation_algorithm in ['word2vec-sg', 'word2vec-cbow']:
            pass

        elif self.representation_algorithm == 'bert':
            return
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







