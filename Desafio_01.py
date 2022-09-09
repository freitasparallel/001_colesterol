#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 21:19:10 2022

@author: jose
"""
import os
import pandas as pd
import numpy as np
import plotly.express as px  #Criação de graficos dinâmnicos
import plotly.graph_objects as go #Para criação e concatenização de graficos
from sklearn.preprocessing import StandardScaler # para relizar a padronização dos dados 
from sklearn.cluster import KMeans
import seaborn as sns

import plotly.io as io
io.renderers.default='browser'
io.renderers.default='svg'

# necessário para gerar as visualizações
%matplotlib inline


os.chdir('/home/jose/Documentos/Pos_Ciencia_de_Dados/Arquiteto_Big_Data/Modulo_1/Desafio/')

clientes = pd.read_excel('dados_clientes.xlsx', sheet_name='Planilha1')
idade = pd.read_csv('idade_clientes.csv', sep = ';')
uf = pd.read_csv('estados_brasileiros.csv', sep = ';')

clientes = clientes.dropna()
idade = idade .dropna()
uf = uf.dropna()

clientes.isna().sum()
idade.isna().sum()
uf.isna().sum()

uf.isna().any()
uf[uf['pais'].isna()]

clientes = clientes[clientes['id_estado'] != 24]
clientes = clientes[clientes['id_estado'] != 18]

clientes = clientes.drop_duplicates()
clientes.duplicated()
clientes.info()

dados_clientes = pd.merge(left = clientes, right = idade ,how='left',on=['id_cliente'], indicator=True)
#dados_clientes = dados_clientes.rename(columns={'_merge': 'merge_idade'})

dados_clientes['merge_idade'].unique()
dados_clientes['_merge'].unique()


del dados_clientes['merge_idade']
del dados_clientes['_merge']
del dados_clientes['id_estado']

#dados_clientes = dados_clientes[dados_clientes['_merge'] == 'both']
dados_clientes = pd.merge(left = dados_clientes, right = uf ,how='left',on=['id_estado'], indicator=True)

dados_clientes.head()

dados_clientes[['peso','colesterol']].head()



idade[idade['id_cliente'] == 63]
clientes[clientes['id_cliente'] == 63]
clientes[clientes['id_cliente'] == 215]
dados_clientes[dados_clientes['id_cliente'] == 63]

idade = idade[idade['id_cliente'] != 63]
idade = idade[idade['id_cliente'] != 215]

clientes[clientes['id_estado'] == 18]
dados_clientes = dados_clientes[dados_clientes['id_estado'] != 18]
dados_clientes = dados_clientes[dados_clientes['id_estado'] != 24]


idade = idade.drop_duplicates()

sns.histplot(data = idade, x= 'idade', bins = 30)



dados_clientes = pd.merge(left = dados_clientes, right = uf, how='inner', on = ['id_estado'])
clientes = pd.merge(left = clientes, right = uf, how='inner', on = ['id_estado'])

dados_clientes = t.copy()
dados_clientes.isna().sum()
dados_clientes.info()
dados_clientes = dados_clientes.dropna()
dados_clientes = dados_clientes.drop_duplicates()
dados_clientes.iloc[:,[1,2]].values

X_saude = dados_clientes[['peso','colesterol']]
X_saude.describe()

X_saude = dados_clientes.iloc[:,[2,3]].values
X_saude = dados_clientes.iloc[:,[1,2 ]].values
X_saude[:10]

normalizar_dados = StandardScaler()
X_saude = normalizar_dados.fit_transform(X_saude)
dados_clientes_t = normalizar_dados.fit_transform(X_saude)

X_saude[:10]

wcss_saude = [] # Cria uma lista vazia
for i in range(1,11):
  kmeans_saude = KMeans(n_clusters=i, random_state=0) # Executa o kmeans para todos os clusters e random_state = 0  para fixar e obter os mesmos resultados
  kmeans_saude.fit(X_saude) # realiza o treinamento
  wcss_saude.append(kmeans_saude.inertia_) # adiciona na lista os valores de wcss


#Visualizando os valores de wcss
for i in range(len(wcss_saude)):
  print('Cluster:', i ,'- Valor do wcss:', wcss_saude[i] )

"""# Criando gráfico para melhor visualização"""
sns.lineplot(data = wcss_saude)

grafico_cotovelo_saude = px.line( x= range(1,11), y=wcss_saude)
grafico_cotovelo_saude.show()

kmeans_saude = KMeans(n_clusters=4, random_state=0)
label_cluster_saude = kmeans_saude.fit_predict(X_saude)

kmeans_saude = KMeans(n_clusters=4, random_state=0)
label_cluster_saude = kmeans_saude.fit_predict(dados_clientes)

#Verifica a classificação dos clusters
label_cluster_saude
kmeans_saude.get_feature_names_out()
kmeans_saude.cluster_centers_
centroides_saude = kmeans_saude.cluster_centers_
centroides_saude

dados_clientes_k = dados_clientes.copy()
dados_clientes_k['clusters'] = label_cluster_saude
dados_clientes_k

1. Alto Risco;
2. Risco Moderado alto;
3. Risco Moderado baixo;
4. Baixo Risco.

a_trocar = {
    2 : 'Baixo Risco', 
    1 : 'Risco Moderado baixo', 
    3 : 'Risco Moderado alto', 
    0 : 'Alto Risco'
}

dados_clientes_k.clusters = dados_clientes_k.clusters.map(a_trocar)
dados_clientes_k.head()

dados_clientes_k = pd.merge(left  = dados_clientes_k, right=uf,on='id_estado')

dados_clientes_k.groupby(['clusters'])['estado'].describe()
sns.histplot(data = dados_clientes_k, y = 'genero', x = 'clusters')

dados_clientes_k.groupby(['clusters'])['estado'].describe()
dados_clientes_k.groupby(['clusters'])['idade'].describe()
dados_clientes_k.groupby(['clusters'])['colesterol'].describe()
dados_clientes_k.groupby(['clusters'])['peso'].describe()
dados_clientes_k.groupby(['clusters','genero'])['colesterol'].describe()
dados_clientes_k.groupby(['clusters','genero'])['colesterol'].describe()

dados_clientes_k.groupby(['clusters'])

"""# Gráfico de agrupamento das características do tamanho e comprimento das pétalas"""

X_saude

grafico_saude = px.scatter(x = X_saude['peso'], y = X_saude['colesterol'], color= label_cluster_saude,hover_name=label_cluster_saude)
grafico_centroide_saude = px.scatter(x = centroides_saude[:,0], y = centroides_saude[:,1],labels=label_cluster_saude)
grafico_final_saude = go.Figure(data = grafico_saude.data + grafico_centroide_saude.data)
grafico_final_saude.show()



fig = px.scatter_3d(dados_clientes_k, x = 'colesterol', y='peso', color='clusters', z='idade')#, opacity = 0.8, size='Age', size_max=30)
fig = px.scatter_3d(dados_clientes_k, x = 'colesterol', y='peso', color='clusters', z='estado')#, opacity = 0.8, size='Age', size_max=30)
fig.show()

grafico_saude = px.scatter(x = dados_clientes_k['peso'], y = dados_clientes_k['colesterol'], color= dados_clientes_k['clusters'])
grafico_centroide_saude = px.scatter(x = centroides_saude[:,0], y = centroides_saude[:,1],labels=label_cluster_saude, size = [7, 7,7,7])
grafico_final_saude = go.Figure(data = grafico_saude.data + grafico_centroide_saude.data)
grafico_final_saude.show()

sns.scatterplot(data = dados_clientes_k, x = dados_clientes_k['peso'], y = dados_clientes_k['colesterol'],hue=dados_clientes_k['clusters'],)
sns.scatterplot(data = dados_clientes_k, x = dados_clientes_k['genero'], y = dados_clientes_k['estado'],hue=dados_clientes_k['clusters'],)
sns.scatterplot(x = centroides_saude[:,0], y = centroides_saude[:,1])
