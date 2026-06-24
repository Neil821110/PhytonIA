# -*- coding: utf-8 -*-
"""
Análisis de base de datos transaccionales - Online Retail
Actividad evaluativa Unidad 4 - Aprendizaje No Supervisado
Maestría en Inteligencia Artificial - Universidad de La Sabana

Este script sigue la misma metodología y estilo de trabajo usados en
Caso 9 (Segmentación de clientes con clustering) y Caso 10 (Clustering
de perfil de consumo energético + detección de anomalías con Isolation Forest).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

import os
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")

# ==========================================================
# 1. CARGA Y EXPLORACIÓN DE LA BASE DE DATOS
# ==========================================================

print("="*70)
print("1. CARGA Y EXPLORACIÓN DE LA BASE DE DATOS")
print("="*70)

df = pd.read_excel("data/Online_Retail.xlsx")
print("Dimensiones de la base de datos:", df.shape)
print(df.head())

print("\nTipos de datos:")
print(df.dtypes)

print("\nEstadísticas descriptivas:")
print(df.describe())

print("\nValores nulos por columna:")
print(df.isna().sum())

# Las columnas Description y CustomerID tienen valores faltantes.
# CustomerID es indispensable para nuestro análisis (segmentación por cliente),
# así que las filas sin CustomerID se eliminarán en el preprocesamiento.

print("\nNúmero de clientes únicos:", df["CustomerID"].nunique())
print("Número de países:", df["Country"].nunique())
print("Número de facturas únicas:", df["InvoiceNo"].nunique())

# ==========================================================
# 2. LIMPIEZA Y PREPROCESAMIENTO
# ==========================================================

print("\n" + "="*70)
print("2. LIMPIEZA Y PREPROCESAMIENTO DE LOS DATOS")
print("="*70)

# Identificamos las cancelaciones: InvoiceNo que empieza con 'C'
df["Cancelado"] = df["InvoiceNo"].astype(str).str.startswith("C")
print("Número de líneas de cancelación:", df["Cancelado"].sum())
print("Porcentaje de cancelaciones:", round(df["Cancelado"].mean()*100, 2), "%")

# Eliminamos filas sin CustomerID (no se pueden asignar a ningún cliente)
df_clean = df.dropna(subset=["CustomerID"]).copy()
print("\nFilas después de eliminar CustomerID nulo:", df_clean.shape[0])

# Eliminamos las cancelaciones para el análisis de comportamiento de compra
# (estas transacciones representan devoluciones, no compras reales)
df_clean = df_clean[~df_clean["Cancelado"]]
print("Filas después de eliminar cancelaciones:", df_clean.shape[0])

# Revisamos cantidades y precios no positivos (probablemente errores o ajustes)
print("\nQuantity <= 0:", (df_clean["Quantity"] <= 0).sum())
print("UnitPrice <= 0:", (df_clean["UnitPrice"] <= 0).sum())

df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
print("Filas después de eliminar cantidades/precios no positivos:", df_clean.shape[0])

# Creamos la variable de valor monetario por línea de producto
df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]

print("\nBase de datos limpia:")
print(df_clean.describe())

# ==========================================================
# 3. CREACIÓN DE LA BASE DE DATOS DE CLIENTES (RFM)
# ==========================================================

print("\n" + "="*70)
print("3. CREACIÓN DE BASE DE DATOS DE PROPIEDADES DE CLIENTES (RFM)")
print("="*70)

# Siguiendo la misma lógica del Caso 9 (creación de una tabla por cliente),
# construimos las variables clásicas de segmentación RFM:
# - Recency: días desde la última compra
# - Frequency: número de facturas distintas
# - Monetary: total gastado

fecha_referencia = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)
print("Fecha de referencia para el cálculo de Recency:", fecha_referencia)

clientes_df = df_clean.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (fecha_referencia - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum")
)

# Agregamos también el número total de artículos comprados y el ticket promedio,
# como hicimos en el Caso 9 al construir características adicionales del cliente
clientes_df["TotalItems"] = df_clean.groupby("CustomerID")["Quantity"].sum()
clientes_df["AvgTicket"] = clientes_df["Monetary"] / clientes_df["Frequency"]

print("\nBase de datos de clientes (RFM):")
print(clientes_df.head())
print("\nEstadísticas descriptivas de la base de clientes:")
print(clientes_df.describe())

# Matriz de correlación entre las variables de clientes
plt.figure(figsize=(6, 5))
sns.heatmap(clientes_df.corr(numeric_only=True), annot=True, vmin=-1, vmax=1,
            cmap="coolwarm", fmt=".2f")
plt.title("Correlación entre variables RFM por cliente")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_correlacion_rfm.png", dpi=150)
plt.close()

# Distribución de las variables (en escala log para Monetary y Frequency,
# dado que el gasto en e-commerce suele tener una distribución muy sesgada)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
sns.histplot(clientes_df["Recency"], ax=axes[0], color="#4C72B0")
axes[0].set_title("Recency (días)")
sns.histplot(clientes_df["Frequency"], ax=axes[1], color="#55A868")
axes[1].set_title("Frequency (facturas)")
sns.histplot(clientes_df["Monetary"], ax=axes[2], color="#C44E52")
axes[2].set_title("Monetary (libras)")
sns.histplot(clientes_df["AvgTicket"], ax=axes[3], color="#8172B2")
axes[3].set_title("Ticket promedio (libras)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_distribuciones_rfm.png", dpi=150)
plt.close()

# ==========================================================
# 4. DETECCIÓN DE ANOMALÍAS A NIVEL DE TRANSACCIÓN
# ==========================================================

print("\n" + "="*70)
print("4. DETECCIÓN DE ANOMALÍAS A NIVEL DE TRANSACCIÓN (ISOLATION FOREST)")
print("="*70)

# Siguiendo la misma lógica del Caso 10, usamos Isolation Forest para
# identificar transacciones (líneas de factura) atípicas, usando como
# variables la cantidad comprada, el precio unitario y el valor total de la línea.

trans_features = df_clean[["Quantity", "UnitPrice", "TotalPrice"]].copy()

# Igual que en el Caso 10, escalamos las variables antes de entrenar el modelo
scaler_trans = StandardScaler()
trans_scaled = scaler_trans.fit_transform(trans_features)

iso_trans = IsolationForest(contamination=0.02, random_state=42)
iso_trans.fit(trans_scaled)
trans_scores = iso_trans.decision_function(trans_scaled)

df_clean["AnomalyScore_Transaccion"] = trans_scores
df_clean["EsAnomalaTransaccion"] = trans_scores < 0

print("Número de transacciones atípicas detectadas:", df_clean["EsAnomalaTransaccion"].sum())
print("Número total de transacciones:", len(df_clean))
print("Porcentaje:", round(df_clean["EsAnomalaTransaccion"].mean()*100, 2), "%")

# Visualizamos las transacciones normales vs atípicas (en escala logarítmica,
# ya que hay un par de valores extremos que dominarían el gráfico en escala normal)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
normales = df_clean[~df_clean["EsAnomalaTransaccion"]]
atipicas = df_clean[df_clean["EsAnomalaTransaccion"]]
ax.scatter(normales["Quantity"], normales["TotalPrice"], alpha=0.15, s=8,
           color="steelblue", label="Transacción normal")
ax.scatter(atipicas["Quantity"], atipicas["TotalPrice"], alpha=0.6, s=14,
           color="crimson", label="Transacción atípica")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Cantidad comprada (Quantity, escala log)")
ax.set_ylabel("Valor total de la línea (TotalPrice, escala log)")
ax.set_title("Transacciones atípicas detectadas con Isolation Forest")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_anomalias_transacciones.png", dpi=150)
plt.close()

print("\nEjemplos de transacciones atípicas (las de mayor valor):")
print(atipicas.sort_values("TotalPrice", ascending=False)
      [["InvoiceNo", "Description", "Quantity", "UnitPrice", "TotalPrice", "CustomerID", "Country"]]
      .head(10))

# ==========================================================
# 5. DETECCIÓN DE CLIENTES ATÍPICOS (ISOLATION FOREST)
# ==========================================================

print("\n" + "="*70)
print("5. DETECCIÓN DE CLIENTES ATÍPICOS (ISOLATION FOREST)")
print("="*70)

# Repetimos el mismo procedimiento del Caso 10 pero ahora a nivel de cliente,
# usando las variables RFM construidas en el paso 3.

scaler_cli_iso = StandardScaler()
clientes_scaled_iso = scaler_cli_iso.fit_transform(clientes_df)

iso_clientes = IsolationForest(contamination=0.05, random_state=42)
iso_clientes.fit(clientes_scaled_iso)
cliente_scores = iso_clientes.decision_function(clientes_scaled_iso)

clientes_df["AnomalyScore"] = cliente_scores
clientes_df["EsClienteAtipico"] = cliente_scores < 0

print("Número de clientes atípicos detectados:", clientes_df["EsClienteAtipico"].sum())
print("Número total de clientes:", len(clientes_df))

print("\nClientes atípicos (ordenados por gasto total):")
print(clientes_df[clientes_df["EsClienteAtipico"]]
      .sort_values("Monetary", ascending=False)
      [["Recency", "Frequency", "Monetary", "AvgTicket"]].head(10))

# Visualizamos Frequency vs Monetary, resaltando los clientes atípicos
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
normales_cli = clientes_df[~clientes_df["EsClienteAtipico"]]
atipicos_cli = clientes_df[clientes_df["EsClienteAtipico"]]
ax.scatter(normales_cli["Frequency"], normales_cli["Monetary"], alpha=0.4, s=20,
           color="steelblue", label="Cliente normal")
ax.scatter(atipicos_cli["Frequency"], atipicos_cli["Monetary"], alpha=0.8, s=30,
           color="crimson", label="Cliente atípico")
ax.set_xlabel("Frequency (número de facturas)")
ax.set_ylabel("Monetary (gasto total en libras)")
ax.set_title("Clientes atípicos detectados con Isolation Forest")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_clientes_atipicos.png", dpi=150)
plt.close()

# Para el proceso de segmentación (clustering) excluimos los clientes atípicos,
# tal como se hizo en el Caso 10 al retirar los días anómalos antes de aplicar K-means
clientes_seg = clientes_df[~clientes_df["EsClienteAtipico"]].copy()
print("\nClientes que quedan para el proceso de segmentación:", len(clientes_seg))

# ==========================================================
# 6. ESTANDARIZACIÓN Y SELECCIÓN DEL NÚMERO DE CLÚSTERES
# ==========================================================

print("\n" + "="*70)
print("6. ESTANDARIZACIÓN Y SELECCIÓN DEL NÚMERO ÓPTIMO DE CLÚSTERES")
print("="*70)

vars_segmentacion = ["Recency", "Frequency", "Monetary", "AvgTicket"]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clientes_seg[vars_segmentacion])

# Método del codo (igual que en el Caso 9)
inertia = []
for i in tqdm(range(1, 11)):
    kmeans = KMeans(n_clusters=i, n_init="auto", random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Método del Codo para Determinar el Número Óptimo de Clústeres")
plt.xlabel("Número de Clústeres")
plt.ylabel("Inercia")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_metodo_codo.png", dpi=150)
plt.close()

# Índice de Calinski-Harabasz (igual que en el Caso 9)
CH_scores = []
for i in tqdm(range(2, 11)):
    kmeans = KMeans(n_clusters=i, n_init="auto", random_state=42)
    kmeans.fit(scaled_data)
    score = calinski_harabasz_score(scaled_data, kmeans.labels_)
    CH_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), CH_scores, marker="o")
plt.title("Índice Calinski-Harabasz para Determinar el Número Óptimo de Clústeres")
plt.xlabel("Número de Clústeres")
plt.ylabel("Índice Calinski-Harabasz")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_calinski_harabasz.png", dpi=150)
plt.close()

# Coeficiente de silueta (igual que en el Caso 10)
sil_scores = []
for i in tqdm(range(2, 11)):
    kmeans = KMeans(n_clusters=i, n_init="auto", random_state=42)
    labels_temp = kmeans.fit_predict(scaled_data)
    sil_scores.append(silhouette_score(scaled_data, labels_temp))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil_scores, marker="o", color="darkorange")
plt.title("Coeficiente de Silueta para Determinar el Número Óptimo de Clústeres")
plt.xlabel("Número de Clústeres")
plt.ylabel("Coeficiente de Silueta")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_silueta.png", dpi=150)
plt.close()

print("Inercia por k:", [round(x, 1) for x in inertia])
print("Calinski-Harabasz por k (desde k=2):", [round(x, 1) for x in CH_scores])
print("Silueta por k (desde k=2):", [round(x, 3) for x in sil_scores])

# Los tres métodos se revisan en conjunto para elegir el número final de clústeres (k).
# Para este conjunto de datos, k=4 ofrece un buen balance entre los tres criterios
# y produce segmentos interpretables desde el punto de vista de negocio.
K_OPTIMO = 4
print(f"\nNúmero de clústeres seleccionado: k = {K_OPTIMO}")

# ==========================================================
# 7. CLUSTERING CON K-MEANS
# ==========================================================

print("\n" + "="*70)
print(f"7. CLUSTERING CON K-MEANS (k={K_OPTIMO})")
print("="*70)

kmeans_final = KMeans(n_clusters=K_OPTIMO, n_init="auto", random_state=42)
clientes_seg["Cluster"] = kmeans_final.fit_predict(scaled_data)

print("Número de clientes por clúster:")
print(clientes_seg["Cluster"].value_counts().sort_index())

print("\nPromedio de cada variable por clúster:")
resumen_clusters = clientes_seg.groupby("Cluster")[vars_segmentacion].mean().round(1)
print(resumen_clusters)

# Boxplots por variable y clúster (mismo estilo que el Caso 9)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, var in zip(axes.flatten(), vars_segmentacion):
    sns.boxplot(x="Cluster", y=var, data=clientes_seg, hue="Cluster", legend=False, ax=ax)
    ax.set_title(f"Distribución de {var} por clúster")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_boxplots_clusters.png", dpi=150)
plt.close()

# Visualización con PCA en 3 componentes (mismo estilo que el Caso 9)
pca = PCA()
pca.fit(scaled_data)
print("\nVarianza explicada acumulada por componente:",
      np.round(pca.explained_variance_ratio_.cumsum(), 3))

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
          pca.explained_variance_ratio_.cumsum(), marker="o")
plt.title("Varianza Acumulada Explicada por los Componentes Principales")
plt.xlabel("Número de Componentes Principales")
plt.ylabel("Varianza Acumulada Explicada")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_pca_varianza.png", dpi=150)
plt.close()

pca3 = PCA(n_components=3)
pca_data = pca3.fit_transform(scaled_data)

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(projection="3d")
scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
                      c=clientes_seg["Cluster"], cmap="viridis", alpha=0.6, s=15)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Visualización 3D de los clústeres en componentes principales")
legend1 = ax.legend(*scatter.legend_elements(), title="Clúster", loc="upper right")
ax.add_artist(legend1)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_pca_3d_clusters.png", dpi=150)
plt.close()

# Gráfico 2D Frequency vs Monetary coloreado por clúster (más fácil de interpretar
# para el lector de negocio que el espacio de componentes principales)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=clientes_seg, x="Frequency", y="Monetary", hue="Cluster",
                 palette="viridis", alpha=0.7, s=40)
plt.title("Segmentos de clientes: Frequency vs Monetary")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/11_clusters_frequency_monetary.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=clientes_seg, x="Recency", y="Monetary", hue="Cluster",
                 palette="viridis", alpha=0.7, s=40)
plt.title("Segmentos de clientes: Recency vs Monetary")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/12_clusters_recency_monetary.png", dpi=150)
plt.close()

# Tamaño relativo de cada clúster
plt.figure(figsize=(6, 6))
clientes_seg["Cluster"].value_counts().sort_index().plot(
    kind="pie", autopct="%1.1f%%", colors=sns.color_palette("viridis", K_OPTIMO))
plt.title("Proporción de clientes por clúster")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/13_proporcion_clusters.png", dpi=150)
plt.close()

# ==========================================================
# 8. ANÁLISIS POR PAÍS (CONTEXTO ADICIONAL DE NEGOCIO)
# ==========================================================

print("\n" + "="*70)
print("8. CONTEXTO ADICIONAL: VENTAS POR PAÍS")
print("="*70)

ventas_pais = df_clean.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
print(ventas_pais.head(10))

plt.figure(figsize=(10, 6))
ventas_pais.head(10).plot(kind="barh", color="teal")
plt.title("Top 10 países por valor total de ventas")
plt.xlabel("Ventas totales (libras esterlinas)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/14_ventas_por_pais.png", dpi=150)
plt.close()

# ==========================================================
# 9. GUARDADO DE RESULTADOS
# ==========================================================

print("\n" + "="*70)
print("9. GUARDANDO RESULTADOS")
print("="*70)

clientes_df_export = clientes_df.merge(
    clientes_seg[["Cluster"]], left_index=True, right_index=True, how="left"
)
clientes_df_export.to_csv("resultados_clientes.csv")
print("Archivo 'resultados_clientes.csv' guardado con las variables RFM, el resultado")
print("de la detección de anomalías y la asignación de clúster para cada cliente.")

print("\nProceso de análisis finalizado correctamente.")
