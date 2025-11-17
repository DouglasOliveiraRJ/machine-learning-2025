# %%

import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx")
df
# %%

from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)

# %%

y = df['Fruta']
caracteristicas = ["Arredondada", "Suculenta", "Vermelha", "Doce"]
X = df[caracteristicas]

# %%
# aprendizado de máquina. Ajuste do modelo

arvore.fit(X, y)

# %%
arvore.predict([[0,0,0,0]])

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(arvore, # é o modelo de árvore (ex: DecisionTreeClassifier).
               feature_names=caracteristicas, # nomes das colunas (as variáveis usadas).
               class_names=arvore.classes_, # nomes das classes (as categorias que a árvore tenta prever).
               filled=True) # preenche os blocos da árvore com cores que ajudam a visualizar as decisões.

# %%

proba = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(proba, index=arvore.classes_)
# %%
