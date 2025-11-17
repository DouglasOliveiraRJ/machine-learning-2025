# %% 

import pandas as pd

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

df = pd.read_csv("../data/abt_churn.csv")
df.head()

# %%

oot = df[df["dtRef"]==df['dtRef'].max()].copy()

# %%

df_train = df[df["dtRef"]<df['dtRef'].max()].copy()

# %%

# Variáveis
features = df_train.columns[2:-1]

# Target
target = 'flagChurn'

X, y = df_train[features], df_train[target]

# %%
# SAMPLE

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                        random_state=42,
                                                                        test_size=0.2,
                                                                        stratify=y,
                                                                        )

# %%

print("Taxa variável resposta Treino", y_train.mean())
print("Taxa variável resposta Test", y_test.mean())

# %%
# EXPLORE (EDA - Análise exploratória)

X_train.isna().sum()

# %%

df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg(["mean", "median"]).T

# diferença absoluta
sumario['diff_abs'] = sumario[0] - sumario[1]

# diferença relativa
sumario['diff_rel'] = sumario[0] / sumario[1]
sumario.sort_values(by=['diff_rel'], ascending=False)

# %%

# arvore de decisão
from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

# %%

pd.Series(arvore.feature_importances_, index=X_train.columns).sort_values(ascending=False)
