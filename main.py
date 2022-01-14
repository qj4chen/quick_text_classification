import pandas as pd
import numpy as np
from classification import QuickTextClassification

df = pd.read_excel('./data/基金基础信息表.xlsx')
df.replace(r'\N', np.nan, inplace=True)
a = QuickTextClassification(dataframe=df, representation_algorithm='bert')
a.determine_optimal_num_clusters(2, 2)
print(a.text)
