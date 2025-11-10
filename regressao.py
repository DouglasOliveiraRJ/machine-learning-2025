# %%

import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()
# %%

from sklearn import linear_model

X = df[['cerveja']] # isso é uma matriz (dataframe)
y = df['nota']      # isso é um vetor (séries)

# Isso é o aprendizado de máquina
reg = linear_model.LinearRegression()
reg.fit(X, y)

# %%

a, b = reg.intercept_, reg.coef_[0]
print(a, b)

# %%

predict = reg.predict(X.drop_duplicates())

# %%

import matplotlib.pyplot as plt

# plot do Gráfico
plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title('Relação Cerveja vs Nota')
plt.xlabel('Cerveja')
plt.ylabel('Nota')

# Reta
plt.plot(X.drop_duplicates(), predict)

plt.legend(['Observado', f'y = {a:.3f} + {b:.3f} x'])

# %%
