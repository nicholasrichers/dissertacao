from mlfinlab.cross_validation import CombinatorialPurgedKFold
import numpy as np
import pandas as pd

#importando os dados
path = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz'
df_training = pd.read_csv(path)
features = [c for c in df_training if c.startswith("feature")]

#tamanho de cada grupo
cdf = df_training.groupby('era').agg(['count'])
group = cdf[cdf.columns[0]].values

last_ix = []
for i,len_era in enumerate(group):
  last_ix_era = sum(group[:(i+1)])
  last_ix.extend(np.array([last_ix_era]*len_era))

#Serie contendo o último indice da respectiva era
samples_info_sets = pd.Series(index= df_training.index, data=np.array(last_ix)-1)

#criando o CV (6,2) como no exemplo do livro
CV = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, samples_info_sets=samples_info_sets)

#criando o split, total de 15 divisoes
cpcv_grp = list(CV.split(X = df_training[features], y = df_training['target'],  groups = df_training.era))


print(len(cpcv_grp))

cpcv_grp