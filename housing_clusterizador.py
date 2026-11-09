import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans #Algoritmo de clusterização
from scipy.spatial.distance import cdist #Cálculo das distâncias até os centroides
from pickle import dump

#1 CARREGAR OS DADOS
dados = pd.read_csv('HousingData.csv', sep=',')

#Separar os atributos categóricos dos quantitativos
#CHAS: 1 se o terreno margeia o rio Charles, 0 caso contrário
#RAD: índice de acessibilidade a rodovias radiais (código, não é uma medida)
colunas_categoricas = ['CHAS', 'RAD']
#MEDV (valor mediano do imóvel) é mantido como atributo: o preço é uma característica
#relevante da residência e ajuda a interpretar os grupos (ex.: grupo de imóveis caros/baratos)
colunas_numericas = dados.drop(columns=colunas_categoricas).columns

#2 TRATAR OS DADOS FALTANTES (NA)
#Numéricos recebem a mediana, categóricos recebem a moda
for coluna in colunas_numericas:
    dados[coluna] = dados[coluna].fillna(dados[coluna].median())
for coluna in colunas_categoricas:
    dados[coluna] = dados[coluna].fillna(dados[coluna].mode()[0]).astype(int)

#Não há necessidade de balancear a base (clusterização não possui classe)

#3 NORMALIZAR OS DADOS
#3.1 Dados numéricos: MinMaxScaler (escala 0 a 1)
dados_numericos = dados[colunas_numericas]
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_numericos)
#3.2 Salvar o modelo normalizador para uso posterior
dump(modelo_normalizador, open('normalizador_housing.model', 'wb'))
#3.3 Normalizar os dados numéricos
dados_numericos_normalizados = modelo_normalizador.transform(dados_numericos)
dados_numericos_normalizados = pd.DataFrame(dados_numericos_normalizados, columns=colunas_numericas)

#3.4 Dados categóricos: one-hot encoding (variáveis dummy 0/1, já na mesma escala do MinMax)
dados_categoricos_normalizados = pd.get_dummies(dados[colunas_categoricas].astype(str), dtype=int)
#3.5 Salvar as colunas geradas para reproduzir a mesma estrutura na inferência
dump(list(dados_categoricos_normalizados.columns), open('colunas_categoricas_housing.model', 'wb'))

#3.6 Juntar numéricos e categóricos normalizados
dados_normalizados = dados_numericos_normalizados.join(dados_categoricos_normalizados)

#4 DETERMINAR O NÚMERO ÓTIMO DE GRUPOS (MÉTODO DO COTOVELO)
#4.1 Calcular a distorção para cada quantidade de clusters
#Distorção = média das distâncias euclidianas de cada residência até o centroide mais próximo
distorcoes = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(dados_normalizados)
    distancias_centroides = cdist(dados_normalizados, kmeans.cluster_centers_, 'euclidean')
    distorcoes.append(sum(np.min(distancias_centroides, axis=1)) / dados_normalizados.shape[0])

#4.2 Calcular a distância de cada ponto da curva até a reta que liga o primeiro e o último ponto
#O ponto mais distante da reta é o "cotovelo"
x0, y0 = K[0], distorcoes[0]
x1, y1 = K[-1], distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x, y = K[i], distorcoes[i]
    numerador = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    denominador = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
    distancias.append(numerador / denominador)

numero_otimo_clusters = K[distancias.index(max(distancias))]
print('Número ótimo de clusters:', numero_otimo_clusters)

#4.3 Plotar o gráfico do cotovelo
plt.plot(K, distorcoes, 'bx-')
plt.plot(numero_otimo_clusters, distorcoes[numero_otimo_clusters - 1], 'ro')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Distorção (distância média ao centroide)')
plt.title('Método do cotovelo')
plt.savefig('cotovelo.png')

#5 TREINAR O MODELO DEFINITIVO
housing_kmeans = KMeans(n_clusters=numero_otimo_clusters, random_state=42, n_init=10).fit(dados_normalizados)
#Salvar para uso posterior
dump(housing_kmeans, open('housing_kmeans.model', 'wb'))

print('Quantidade de residências por cluster:')
print(pd.Series(housing_kmeans.labels_).value_counts().sort_index())
