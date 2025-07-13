import pandas as pd
import plotly.express as px
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
x_train, x_test, y_train, y_test = train_test_split(variavel_explicativa, variavel_alvo, stratify=variavel_alvo, random_state=5)
