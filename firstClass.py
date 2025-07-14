import pandas as pd
import plotly.express as px
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('marketing_investimento.csv')

print(df.head())
print(df.info()) # Como não há dados nulos, não há necessidade de tratamento

# Primeiro vamos explorar os dados da variável alvo e categóricos
fig = px.histogram(df, x='aderencia_investimento', text_auto=True)
# fig.show()

fig = px.histogram(df, x='estado_civil', color='aderencia_investimento', barmode='group', text_auto=True)
# fig.show()

fig = px.histogram(df, x='escolaridade', color='aderencia_investimento', barmode='group', text_auto=True)
# fig.show()

fig = px.histogram(df, x='inadimplencia', color='aderencia_investimento', barmode='group', text_auto=True)
# fig.show()

fig = px.histogram(df, x='fez_emprestimo', color='aderencia_investimento', barmode='group', text_auto=True)
# fig.show()

fig = px.box(df, x='idade', color='aderencia_investimento')
# fig.show()

fig = px.box(df, x='saldo', color='aderencia_investimento')
# fig.show()

fig = px.box(df, x='tempo_ult_contato', color='aderencia_investimento')
# fig.show()

fig = px.box(df, x='numero_contatos', color='aderencia_investimento')
# fig.show()

# Separando as variáveis explicativas e a variável alvo
variavel_alvo = df['aderencia_investimento']
variavel_explicativa = df.drop(columns=['aderencia_investimento'], axis=1)

# Aplicando One Hot Encoding nas variáveis categóricas
colunas = variavel_explicativa.columns
one_hot = make_column_transformer(
    (OneHotEncoder(drop='if_binary'), ['estado_civil', 'escolaridade', 'inadimplencia', 'fez_emprestimo']),
    remainder='passthrough',
    sparse_threshold=0
)

variavel_explicativa = one_hot.fit_transform(variavel_explicativa)
print("\nColunas antes do One Hot Encoding:", colunas.tolist())
print("Número de colunas após One Hot Encoding:", one_hot.get_feature_names_out(colunas))

pd_variavel_explicativa = pd.DataFrame(variavel_explicativa, columns=one_hot.get_feature_names_out(colunas))
print(pd_variavel_explicativa.head())

# Utilizando LabelEncoder para a variável alvo, transformando em números inteiros (0 e 1)
label_encoder = LabelEncoder()
variavel_alvo = label_encoder.fit_transform(variavel_alvo)
print(variavel_alvo)

# Separando os dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(pd_variavel_explicativa, variavel_alvo, stratify=variavel_alvo, random_state=5)

# Criando o modelo dummy
# Resultado de 60% de acerto
dummy = DummyClassifier()
dummy.fit(x_train, y_train)

# Avaliando o modelo dummy
print(dummy.score(x_test, y_test))

# Criando o modelo de árvore de decisão
arvore = DecisionTreeClassifier(random_state=5)
arvore.fit(x_train, y_train)

# Previsão de novos dados
print(arvore.predict(x_test))

# Avaliando o modelo de árvore de decisão - Resultado de 66%
print(arvore.score(x_test, y_test))

plt.figure(figsize=(15, 6))
plot_tree(arvore, filled=True, class_names = ['Não', 'Sim'], feature_names=x_train.columns)
# plt.show()

# Criando o modelo de árvore de decisão, foi visto que o modelo anterior apenas decorou os padrões
# Agora iremos diminuir a profundidade da árvore
arvore = DecisionTreeClassifier(max_depth=3, random_state=5)
arvore.fit(x_train, y_train)

# Avaliando o modelo de árvore de decisão - Resultado de 71% > dummy
print(arvore.score(x_test, y_test))

plt.figure(figsize=(15, 6))
plot_tree(arvore, filled=True, class_names = ['Não', 'Sim'], feature_names=x_train.columns)
# plt.show()

# Normalizando os dados para usar no modelo KNN
normalizador = MinMaxScaler()
x_train_normalizado = normalizador.fit_transform(x_train)
print(pd.DataFrame(x_train_normalizado))

# Criando o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_normalizado, y_train)

x_test_normalizado = normalizador.transform(x_test) # Transformando os dados de teste

print(knn.score(x_test_normalizado, y_test)) # 68% de acerto