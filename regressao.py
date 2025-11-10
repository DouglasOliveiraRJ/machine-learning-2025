# %%

import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()
# %%

from sklearn import linear_model
from sklearn import tree

X = df[['cerveja']] # isso é uma matriz (dataframe)
y = df['nota']      # isso é um vetor (séries)

# Isso é o aprendizado de máquina
reg = linear_model.LinearRegression()
reg.fit(X, y)

a, b = reg.intercept_, reg.coef_[0]
print(a, b)

predict_reg = reg.predict(X.drop_duplicates())

# árvore sem limite de nós
arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)

predict_arvore_full = arvore_full.predict(X.drop_duplicates())

# árvore com 4 nós
arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
arvore_d2.fit(X, y)

predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())
# %%

import matplotlib.pyplot as plt

# plot do Gráfico
plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title('Relação Cerveja vs Nota')
plt.xlabel('Cerveja')
plt.ylabel('Nota')

# Reta
plt.plot(X.drop_duplicates(), predict_reg)
plt.plot(X.drop_duplicates(), predict_arvore_full)
plt.plot(X.drop_duplicates(), predict_arvore_d2, color='magenta')

plt.legend(['Observado', 
            f'y = {a:.3f} + {b:.3f} x',
            'Árvore Full',
            'Árvore Depth',
            ])

# %%

# mostrando a árvore d2
plt.figure(dpi=400)

tree.plot_tree(arvore_d2,
               feature_names=['cerveja'],
               filled=True)
# %%
