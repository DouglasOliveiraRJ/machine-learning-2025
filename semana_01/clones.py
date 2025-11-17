# %%

import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")
df.head()

# %%

features = ["Massa(em kilos)", "Estatura(cm)"]
target = ['Status ']

X = df[features]
y = df[target]

X = X.replace({
    "Yoda": 1, "Shaak Ti": 2, "Obi-Wan Kenobi": 3, "Mace Windu": 4, "Aayla Secura": 5,
    "Tipo 1": 1, "Tipo 2": 2, "Tipo 3": 3, "Tipo 4": 4, "Tipo 5": 5,
})

# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)
plt.figure(figsize=(6, 6))

tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True,
               max_depth=3
               )
# %%
