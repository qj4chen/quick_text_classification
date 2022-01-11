# quick_text_classification
旨在提供一个简便的api，根据pd.dataframe, 对其中的文本列进行预处理、聚类、训练分类算法等任务.

目前已经完成的部分：

- [x] 既可自行指定文本列, 也可根据 dtype 自动识别筛选文本列
- [x] 将dataframe中的文本列聚合, 清洗, 包括去除stock_code, url, email, 数字
- [x] 目前已添加的表征算法包括: bag-of-words, tf-idf

目前还未完成的部分:

- [ ] 文本词性识别, 作为额外信息, 补充到tf-idf和bag-of-words等lexicon-level的表征算法中
- [ ] 添加semantic-level的模型, 如:预训练word2vec, word2vec, bert
