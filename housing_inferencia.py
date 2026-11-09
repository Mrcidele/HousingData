#Receber uma nova residência, determinar a qual grupo ela pertence e exibir o centroide desse grupo

import pandas as pd
from pickle import load

colunas_categoricas = ['CHAS', 'RAD']

#1 RECEBER OS DADOS DA NOVA RESIDÊNCIA
#Valores de exemplo usados quando o usuário apenas pressiona Enter
exemplo = {
    'CRIM': 0.05, 'ZN': 20, 'INDUS': 4.5, 'CHAS': 0, 'NOX': 0.45,
    'RM': 6.8, 'AGE': 40, 'DIS': 5.5, 'RAD': 4, 'TAX': 300,
    'PTRATIO': 17.0, 'B': 390, 'LSTAT': 6.0, 'MEDV': 28.0
}
print('Informe os dados da nova residência (Enter usa o valor de exemplo):')
nova_residencia = {}
for atributo, valor_padrao in exemplo.items():
    valor = input(f'{atributo} [{valor_padrao}]: ').strip().replace(',', '.')
    nova_residencia[atributo] = float(valor) if valor else valor_padrao
for coluna in colunas_categoricas:
    nova_residencia[coluna] = int(nova_residencia[coluna])
nova_residencia = pd.DataFrame([nova_residencia])

#2 NORMALIZAR A NOVA RESIDÊNCIA
#2.1 Carregar o normalizador e as colunas categóricas salvos no treinamento
normalizador = load(open('normalizador_housing.model', 'rb'))
colunas_dummies = load(open('colunas_categoricas_housing.model', 'rb'))
colunas_numericas = normalizador.feature_names_in_

#2.2 Normalizar os dados numéricos
nova_numericos = normalizador.transform(nova_residencia[colunas_numericas])
nova_numericos = pd.DataFrame(nova_numericos, columns=colunas_numericas)

#2.3 Normalizar os dados categóricos (mesmas colunas dummy do treinamento)
nova_categoricos = pd.get_dummies(nova_residencia[colunas_categoricas].astype(str), dtype=int)
nova_categoricos = nova_categoricos.reindex(columns=colunas_dummies, fill_value=0)

nova_residencia_normalizada = nova_numericos.join(nova_categoricos)

#3 INFERIR O GRUPO
housing_kmeans = load(open('housing_kmeans.model', 'rb'))
grupo = housing_kmeans.predict(nova_residencia_normalizada)[0]
print('A nova residência pertence ao grupo:', grupo)

#4 EXIBIR O CENTROIDE DESNORMALIZADO
centroide = pd.DataFrame([housing_kmeans.cluster_centers_[grupo]], columns=nova_residencia_normalizada.columns)

#4.1 Numéricos: inverter o MinMaxScaler
centroide_numericos = normalizador.inverse_transform(centroide[colunas_numericas])
centroide_desnormalizado = pd.DataFrame(centroide_numericos, columns=colunas_numericas)

#4.2 Categóricos: a categoria com maior peso no centroide é o valor representativo
for coluna in colunas_categoricas:
    dummies = centroide[[c for c in colunas_dummies if c.startswith(coluna + '_')]]
    centroide_desnormalizado[coluna] = int(dummies.idxmax(axis=1)[0].split('_')[1])

#Reordenar as colunas na ordem original da base
centroide_desnormalizado = centroide_desnormalizado[nova_residencia.columns]
print('Centroide do grupo', grupo, '(desnormalizado):')
for coluna in centroide_desnormalizado.columns:
    valor = centroide_desnormalizado[coluna][0]
    if coluna in colunas_categoricas:
        print(f'{coluna:8} {int(valor)}')
    else:
        print(f'{coluna:8} {round(valor, 4) + 0.0:.4f}')
