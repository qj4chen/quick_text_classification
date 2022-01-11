import string
import re

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer


class TextPreprocessing:
    __lemmatizer = WordNetLemmatizer()
    __contractions_dict = {
        "ain't": "are not",
        "'s": " is",
        "aren't": "are not",
        "isn't": "is not",
        "can't": "can not",
        "'ve": "have",
        "'d": "would",
        "don't": "do not",
        "doesn't": "does not",
        "did": "do",
        "did't": "do not"
    }

    def __init__(self,
                 x,
                 contractions_dict=None,
                 lemmatizer=None,
                 ):
        """
        :param x: the text to preprocess
        :param contractions_dict: the english contraction mapping rules
        :param lemmatizer: the nltk lemmatizer to further process the text
        """
        x = x.strip()
        x = x.lower()
        self.x = x
        if contractions_dict is not None:
            self.contractions_dict = contractions_dict
            self.contractions_re = re.compile('(%s)' % '|'.join(self.contractions_dict.keys()))
        else:
            self.contractions_re = re.compile('(%s)' % '|'.join(self.__contractions_dict.keys()))

        self.x = self.expand_contractions()
        self.x = self.replace_stock_code(self.x)
        self.x = self.replace_email(self.x)
        self.x = self.replace_url(self.x)

        self.x = self.remove_numbers(self.x)
        self.x = self.remove_punctuations(self.x)

        if lemmatizer is not None:
            self.lemmatizer = lemmatizer
            self.x = " ".join([self.lemmatizer.lemmatize(word) for word in self.x.split()])
        else:
            self.x = " ".join([self.__lemmatizer.lemmatize(word) for word in self.x.split()])

    @staticmethod
    def remove_punctuations(x):
        return re.sub('[%s]' % re.escape(string.punctuation), '', x)

    @staticmethod
    def remove_numbers(x):
        return re.sub(r'W*\dw*', '', x)

    @staticmethod
    def replace_stock_code(x):
        return re.sub(r'[0-9]{4,6}\.[A-Za-z]{2} ', 'wind_code ', x)

    @staticmethod
    def replace_email(x):
        return re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', 'emailadd', x)

    @staticmethod
    def replace_url(x):
        return re.sub(r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)',
                      'urladd', x)

    def expand_contractions(self):
        def replace(match):
            return self.contractions_dict[match.group(0)]

        return self.contractions_re.sub(replace, self.x)

    def __repr__(self):
        return self.x

    @classmethod
    def batch_preprocess(cls,
                         x,
                         contractions_dict=None,
                         lemmatizer=None,
                         return_string=False
                         ):
        if return_string:
            return [cls(_, contractions_dict, lemmatizer, return_string).x for _ in x]
        else:
            return [cls(_, contractions_dict, lemmatizer, return_string) for _ in x]


def clean_text(x):
    if isinstance(x, str):
        x = [x]
    elif isinstance(x, pd.Series):
        x = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        raise NotImplementedError
    return np.array([str(TextPreprocessing(_)) for _ in x])



