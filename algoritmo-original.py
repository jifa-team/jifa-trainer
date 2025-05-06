# Exemplo 1

# importa o módulo que contém os conjuntos de dados
from sklearn import datasets as ds
 # carrega o conjunto iris de dados
iris = ds.load_iris()

# Exemplo 2

# importa a biblioteca pandas
import pandas as pa

# cria um DataFrame a partir do objeto íris
df = pa.DataFrame(data=iris['data'], columns=iris['feature_names'])
# adiciona ao DataFrame a coluna de atributos alvo
dfT = df.copy()
dfT['target'] = iris['target'] 
print(df), print(dfT)

# Exemplo 3:
from sklearn import datasets as ds
mb = ds.make_blobs(n_samples=100, n_features=4)

# Exemplo 4

# importação do módulo KMeans da biblioteca SciKit-Learn
from sklearn.cluster import KMeans
# alg recebe o objeto do método KMeans aleatório básico
alg = KMeans(init='random')

# Exemplo 5

# importação do módulo KMeans da biblioteca SciKit-Learn
from sklearn.cluster import KMeans 

# alg recebe o objeto do método KMeans configurado para # identificar 4 clusters. O valor 50 serve de base para o # cálculo do ponto inicial de cada centroide.
alg = KMeans(n_clusters=4, random_state=50)

# Exemplo 6

# carrega bibliotecas e módulos necessários
from sklearn import datasets as ds
from sklearn.cluster import KMeans
import pandas as pa
import seaborn as sb
import matplotlib.pyplot as pp

# distribuição aleatória: 100 objetos, 4 grupos, 2 carac-terísticas desvio padrão de 1,3
X, y = ds.make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=1.3)
# inicializa e treina um algoritmo de K-Means
km = KMeans(n_clusters=4, random_state=50) 
km.fit(X)
prevs = km.predict(X)
# DataFrames para o conjunto de dados original com seu 
# atributo alvo e outro quadro com o atributo alvo previsto
#  # a partir do K-Means
xDF = pa.DataFrame(data=X, columns=['Attr1', 'Attr2'])
pDF = xDF.copy() 
xDF['Target'] = y 
pDF['Target'] = prevs
# divisão da figura em duas colunas para plotar os dois # conjuntos
fig, axes = pp.subplots(1, 2, figsize=(12,4))
sb.scatterplot(data=xDF, x='Attr1', y='Attr2', hue='Target', palette='rainbow', ax=axes[0])
sb.scatterplot(data=pDF, x='Attr1', y='Attr2', hue='Target', palette='rainbow', ax=axes[1])

# Exemplo 7
# importação das bibliotecas e seleção dos módulos a serem # utilizados
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pa
import seaborn as sb
import matplotlib.pyplot as pp 
import numpy as np
# importação e separação dos dados
iDS = load_iris()
iData = iDS['data']
iTarget = iDS['target']
iTarget_Names = iDS['target_names']
iFeature_Names = iDS['feature_names']
# formatação dos dados em quadro da biblioteca pandas para
# facilitar a visualização
iDF = pa.DataFrame(data=iData, columns=iFeature_Names)
iDF_Target = iDF.copy()
iDF_Target['specie'] = iTarget_Names[iTarget]
# visualizar a sobreposição das espécies entre todas as
# combinações de pares de características
sb.pairplot(data=iDF_Target, hue='specie')

# Exemplo 8
# seleção das duas características
iData_SL_PW = iData[:, [0,3]]
iDF_SL_PW = pa.DataFrame(data=iData_SL_PW, columns=[iFeature_Names[0], iFeature_Names[3]]) 
iDF_SL_PW_T = iDF_SL_PW.copy()
iDF_SL_PW_T['specie'] = iDF_Target['specie'] 
# distribuição após seleção dos atributos sem rótulos definidos
prevFeatureNames = iDF_SL_PW_T.keys() 
sb.scatterplot(data=iDF_SL_PW_T, x=prevFeatureNames[0], y=prevFeatureNames[1])

# Exemplo 9

# quantidade de clusters será a mesma da quantidade de espécies do conjunto de dados
k = len(iTarget_Names)
# configuração do algoritmo e treinamento
alg = KMeans(n_clusters=k, random_state=50) 
alg.fit(X=iData_SL_PW)
# construção de novo quadro considerando os agrupamentos previstos pelo K-Means
prevDataFrame = iDF_SL_PW.copy()
prevDataFrame['specie'] = alg.predict(iData_SL_PW)
prevDataFrame
pp.show()

# Exemplo 10
translateDic = { 0: 'virginica', 1: 'setosa', 2: 'versicolor' }
prevDataFrame['specie'] = [translateDic[s] for s in 
                           prevDataFrame['specie']]
fig, axes = pp.subplots(1, 2, figsize=(12,4))
fig1 = sb.scatterplot(data=iDF_SL_PW_T, x=prevFeatureNames[0], y=prevFeatureNames[1], hue='specie', ax=axes[0])
fig2 = sb.scatterplot(data=prevDataFrame, x=prevFeatureNames[0], y=prevFeatureNames[1], hue='specie', ax=axes[1])
pp.show()

# Exemplo 11 len(prevDataFrame[iDF_SL_PW_T['specie'] != prevDataFrame['specie']])
cm = confusion_matrix(iDF_SL_PW_T['specie'], prevDataFrame['specie'])
cmtx = pa.DataFrame( cm, index=('Real: ' + pa.DataFrame(iTarget_Names)) [0], columns=('Prev: ' + pa.DataFrame(iTarget_Names))[0] )
cmtx, accuracy_score(iDF_SL_PW_T['specie'], prevDataFrame['specie'])

# Exemplo 12
from sklearn.metrics import silhouette_score
prevs = alg.predict(iData_SL_PW)
silhouette_score(iData_SL_PW, prevs)

# Exemplo 13
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_score(iData_SL_PW, prevs)


 




