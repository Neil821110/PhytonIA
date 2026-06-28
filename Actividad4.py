#!/usr/bin/env python
# coding: utf-8

# # Desarrollo de Entrega Unidad 4 – Base de Datos de tienda usando las Unidades 9 y 10
# 
# En este caso trabajaremos con una base de datos de órdenes de bases de datos de tienda de la unidad 4.

# Iniciemos importando las librerías necesarias

# In[1]:


#!conda install sklearn.ensemble -y
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go


# # Cargando y Analizando la Base de datos
# Ahora importamos nuestra base de datos

# In[ ]:


consumer_data = pd.read_excel('data/Online Retail.xlsx')
consumer_data


# # Cambiar los nombres de la base de datos creando un directorio 

# In[ ]:


from os import rename
consumer_data.rename(columns={'InvoiceNo':'Numero_factura','StockCode':'Cod_prod','Description':'Nombre_Producto','Quantity':'Cantidad','UnitPrice':'Precio unitario','InvoiceDate':'Fecha_compra','CustomerID':'Cod_Cliente','Country':'País'},inplace=True)
#consumer_data = pd.DataFrame(consumer_dat["Numero_factura"].astype(str))
#consumer_data.to_csv('data/Online_Retail.csv')
consumer_data


# Se analiza la estadistica de las variables númericas

# In[ ]:


consumer_data.describe()


# Tipo de datos que se tienen

# In[ ]:


consumer_data.dtypes


# 

# In[ ]:


print('Cantidad de paises: ', consumer_data['País'].nunique())
print('Cantidad de productos: ', consumer_data['Cod_prod'].nunique())
print('Cantidad de usuarios: ', consumer_data['Cod_Cliente'].nunique())


# In[ ]:


# Gráfica de órdenes por país con nombres en español
paises_ordenes = consumer_data.groupby('País')['Numero_factura'].nunique().sort_values(ascending=False).head(15)

traduccion_paises = {
    'United Kingdom': 'Reino Unido',
    'France': 'Francia',
    'Germany': 'Alemania',
    'Spain': 'España',
    'Poland': 'Polonia',
    'Netherlands': 'Países Bajos',
    'Belgium': 'Bélgica',
    'Switzerland': 'Suiza',
    'Portugal': 'Portugal',
    'Australia': 'Australia',
    'Norway': 'Noruega',
    'EIRE': 'Irlanda',
    'Sweden': 'Suecia',
    'Denmark': 'Dinamarca',
    'Finland': 'Finlandia',
    'Italy': 'Italia',
    'Channel Islands': 'Islas del Canal',
    'Cyprus': 'Chipre',    
}

paises_traducidos = [traduccion_paises.get(p, p) for p in paises_ordenes.index]

plt.figure(figsize=(12, 8))
sns.barplot(x=paises_ordenes.values, y=paises_traducidos, palette='viridis',color='goldenrod')
plt.title('Top 15 países con más órdenes')
plt.xlabel('Número de órdenes')
plt.ylabel('País')
plt.tight_layout()
plt.show()


# In[ ]:


codigos_especiales = ['POST', 'DOT', 'M', 'BANK CHARGES', 'AMAZONFEE', 'CRUK', 'PADS', 'C2']

productos_mas_vendidos = (
    consumer_data[~consumer_data['Cod_prod'].isin(codigos_especiales)]
    .groupby('Nombre_Producto')['Cantidad']
    .sum()
    .sort_values(ascending=False)
    .head(20)
)

fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(x=productos_mas_vendidos.values, y=productos_mas_vendidos.index,
            hue=productos_mas_vendidos.index, palette='coolwarm', legend=False, ax=ax)
ax.bar_label(ax.containers[0], fmt='%.0f', padding=3, fontsize=8)
ax.set_title('Top 20 productos por cantidad vendida')
ax.set_xlabel('Cantidad total vendida')
ax.set_ylabel('Producto')
plt.tight_layout()
plt.show()


# Verifiquemos si tenemos valores faltantes en nuestra base de datos

# In[ ]:


consumer_data.isna().sum()


# # Tratamiento de la base de Datos
# Teniendo en cuenta el código del producto se rellenaran los espacios

# In[ ]:


def completar_nombres_productos(df):
    """
    Rellena los espacios en blanco (valores nulos) de la columna 'Nombre_Producto' 
    buscando el nombre asociado a su 'Cod_prod' en el resto del DataFrame.
    """
    # 1. Creamos un mapeo (diccionario) de Cod_prod -> Nombre_Producto
    # Descartamos los nulos y nos quedamos con la primera aparición de cada producto
    mapeo_nombres = (
        df[['Cod_prod', 'Nombre_Producto']]
        .dropna(subset=['Nombre_Producto'])
        .drop_duplicates(subset=['Cod_prod'])
        .set_index('Cod_prod')['Nombre_Producto']
    )

    # 2. Rellenamos los valores nulos en 'Nombre_Producto' usando la función map() 
    # que busca el Cod_prod en nuestro mapeo
    df['Nombre_Producto'] = df['Nombre_Producto'].fillna(df['Cod_prod'].map(mapeo_nombres))

    return df

# funcion adicional 


# Ejemplo de uso con tu DataFrame:
consumer_data = completar_nombres_productos(consumer_data)

# Para verificar que funcionó (opcional):
print("Valores nulos en 'Nombre_Producto':", consumer_data['Nombre_Producto'].isnull().sum())


# Verificamos el resultado

# In[ ]:


consumer_data.isna().sum()


# In[ ]:


consumer_data


# In[ ]:





# # Tener pendiente consumer_data_a para el analisis de algoritmos no descriptivos

# In[ ]:


# Reemplazar celdas vacías en Cod_Cliente por NaN
consumer_data_a = consumer_data.copy()
consumer_data_a['Cod_Cliente'] = consumer_data_a['Cod_Cliente'].replace(['', ' ', np.nan], np.nan)

# Mostrar el conteo de valores NaN después del reemplazo
print("Valores NaN después del reemplazo:")
print(consumer_data_a.isna().sum())
#print("\nValores únicos en Cod_Cliente (primeros 10):")
#print(consumer_data_a['Cod_Cliente'].unique()[:10])


# In[ ]:


# Reemplazar valores vacíos en Cod_Cliente por NaN y eliminar filas de consumer_data
consumer_data['Cod_Cliente'] = consumer_data['Cod_Cliente'].replace(['', ' '], np.nan)

filas_antes = len(consumer_data)
filas_vacias = consumer_data['Cod_Cliente'].isna().sum()
print(f"Número de filas antes de eliminar: {filas_antes}")
print(f"Filas con Cod_Cliente vacío: {filas_vacias}")

# Eliminar filas con Cod_Cliente vacío en consumer_data
consumer_data = consumer_data.dropna(subset=['Cod_Cliente']).reset_index(drop=True)
filas_despues = len(consumer_data)
print(f"\nNúmero de filas después de eliminar: {filas_despues}")
print(f"Filas eliminadas: {filas_antes - filas_despues}")
print(f"Filas vacías restantes en Cod_Cliente: {consumer_data['Cod_Cliente'].isna().sum()}")


# # Eliminar los datos duplicados 

# In[ ]:


print(f"consumer_data — filas antes: {len(consumer_data)}")
print(f"Duplicados exactos encontrados: {consumer_data.duplicated().sum()}")
consumer_data = consumer_data.drop_duplicates().reset_index(drop=True)
print(f"consumer_data — filas después: {len(consumer_data)}")

print(f"\nconsumer_data_a — filas antes: {len(consumer_data_a)}")
print(f"Duplicados exactos encontrados: {consumer_data_a.duplicated().sum()}")
consumer_data_a = consumer_data_a.drop_duplicates().reset_index(drop=True)
print(f"consumer_data_a — filas después: {len(consumer_data_a)}")


# In[ ]:





# # Anexando una columna de estado de la compraer eincluye la columna monto para luego hacer analisisq

# In[ ]:


consumer_data['Monto'] = consumer_data['Cantidad'] * consumer_data['Precio unitario']
consumer_data_a['Monto'] = consumer_data_a['Cantidad'] * consumer_data_a['Precio unitario']

consumer_data[['Cantidad', 'Precio unitario', 'Monto']].describe()


# Anexando una columna de estado de la compra

# In[ ]:


def asignar_estado_compra(df):
    """
    Crea una nueva columna 'Estado de la compra'.
    Si el 'Numero_factura' contiene la letra 'C', se marca como 'Cancelado', 
    de lo contrario se marca como 'Realizado'.
    """
    # 1. Nos aseguramos de tratar la columna Numero_factura como texto (string)
    # 2. Comprobamos si contiene la letra 'C' (case=False para que no distinga entre mayúsculas y minúsculas)
    condicion_cancelado = df['Numero_factura'].astype(str).str.contains('C', case=False, na=False)

    # 3. np.where funciona así: np.where(condicion, valor_si_verdadero, valor_si_falso)
    df['Estado de la compra'] = np.where(condicion_cancelado, 'Cancelado', 'Realizado')

    return df
# Ejemplo de uso con tu DataFrame:
consumer_data = asignar_estado_compra(consumer_data)
# Para verificar que funcionó viendo algunas filas canceladas:
display(consumer_data[consumer_data['Estado de la compra'] == 'Cancelado'].head())


# Se adiciona la linea para trasformar la columna Numero de Factura a tipo texto.  

# In[ ]:


consumer_data["Numero_factura"] = consumer_data["Numero_factura"].astype(str)
consumer_data["Cod_prod"] = consumer_data["Cod_prod"].astype(str)
consumer_data["Nombre_Producto"] = consumer_data["Nombre_Producto"].astype(str)
consumer_data


# Se realiza una la priemra grafica

# se agrega las columnas fecha día mes hora

# In[ ]:


# 2. Creamos las nuevas columnas extrayendo la información
dias_semana = {
    0: 'Lunes', 1: 'Martes', 2: 'Miércoles',
    3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'
}
#consumer_data['Dia_compra'] = consumer_data['Fecha_compra'].dt.dayofweek.map(dias_semana)

consumer_data['Mes_compra'] = consumer_data['Fecha_compra'].dt.month
meses_mapping = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}
consumer_data['Mes_compra'] = consumer_data['Mes_compra'].map(meses_mapping)
consumer_data['Mes_compra'] = pd.Categorical(
    consumer_data['Mes_compra'],
    categories=list(meses_mapping.values()),
    ordered=True
)
consumer_data['Hora_compra'] = consumer_data['Fecha_compra'].dt.hour
# Para verificar que las columnas se crearon correctamente:
display(consumer_data[['Fecha_compra', 'Mes_compra', 'Hora_compra']].head())
consumer_data


# In[ ]:





# In[ ]:


#consumer_data = consumer_data.order(columns={'1': 'Enero', '2': 'Febrero', '3': 'Marzo', '4': 'Abril', '5': 'Mayo', '6': 'Junio', '7': 'Julio', '8': 'Agosto', '9': 'Septiembre', '10': 'Octubre', '11': 'Noviembre', '12': 'Diciembre'})
consumer_data['Mes_compra'].value_counts().sort_index().plot(kind='bar', figsize=(12, 6))


# In[ ]:


consumer_data.describe()


# se vuelve a ver que tipo de datos que tiene la base 

# In[ ]:


consumer_data.dtypes


# 

# In[ ]:


#consumer_data['Dia_compra'].order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
#consumer_data['Dia_compra'].value_counts().reindex(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']).plot(kind='bar', figsize=(12, 6))
consumer_data['Dia_semana'] = consumer_data['Fecha_compra'].dt.day_name()


# In[ ]:


# 1. Crear la columna de día de la semana (en español, ordenada Lunes->Domingo)
dias_es = {0:'Lunes', 1:'Martes', 2:'Miércoles', 3:'Jueves', 4:'Viernes', 5:'Sábado', 6:'Domingo'}
consumer_data['Dia_semana'] = consumer_data['Fecha_compra'].dt.dayofweek.map(dias_es)
orden_dias = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
consumer_data['Dia_semana'] = pd.Categorical(consumer_data['Dia_semana'], categories=orden_dias, ordered=True)

# 2. Agregaciones correctas por día de la semana
items_por_dia = consumer_data.groupby('Dia_semana')['Cantidad'].sum()
ordenes_por_dia = consumer_data.groupby('Dia_semana')['Numero_factura'].nunique()

# 3. Graficar
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

items_por_dia.plot(kind='bar', ax=axes[0], color='goldenrod')
axes[0].set_title('Número de ítems pedidos por día de la semana')
axes[0].set_xlabel('Día de la semana')
axes[0].set_ylabel('Número de ítems')

ordenes_por_dia.plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].set_title('Número de órdenes por día de la semana')
axes[1].set_xlabel('Día de la semana')
axes[1].set_ylabel('Número de órdenes')

plt.tight_layout()
plt.show()


# # Grafico de los paises a los cuales se envia los articulos 

# In[ ]:


# Métrica correcta: facturas únicas, no filas
traduccion_paises = {
    'United Kingdom': 'Reino Unido',
    'France': 'Francia',
    'Germany': 'Alemania',
    'Spain': 'España',
    'Poland': 'Polonia',
    'Netherlands': 'Países Bajos',
    'Belgium': 'Bélgica',
    'Switzerland': 'Suiza',
    'Portugal': 'Portugal',
    'Australia': 'Australia',
    'Norway': 'Noruega',
    'EIRE': 'Irlanda',
    'Sweden': 'Suecia',
    'Denmark': 'Dinamarca',
    'Finland': 'Finlandia',
    'Italy': 'Italia',
    'Channel Islands': 'Islas del Canal',
    'Cyprus': 'Chipre',    
}

ordenes_por_pais = consumer_data.groupby('País')['Numero_factura'].nunique().sort_values(ascending=False)
top15 = ordenes_por_pais.head(15)
top15.index = [traduccion_paises.get(p, p) for p in top15.index]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(top15.index[::-1], top15.values[::-1], color='steelblue')
axes[0].bar_label(axes[0].containers[0], fmt='%.0f', padding=3, fontsize=8)
axes[0].set_xscale('log')

# 
# 

axes[0].set_title('Top 15 países por número de órdenes (escala log)')
axes[0].set_xlabel('Número de órdenes (escala log)')

top15_sin_uk = top15.drop('Reino Unido', errors='ignore')
axes[1].barh(top15_sin_uk.index[::-1], top15_sin_uk.values[::-1], color='seagreen')
axes[1].bar_label(axes[1].containers[0], fmt='%.0f', padding=3)
axes[1].set_title('Top 15 países por número de órdenes (sin Reino Unido)')
axes[1].set_xlabel('Número de órdenes')

plt.tight_layout()
plt.show()


# Analisis de Hora del día

# In[ ]:


# Gráfico de compras por hora del día
compras_por_hora = consumer_data.groupby('Hora_compra')['Numero_factura'].nunique().sort_index()
plt.figure(figsize=(12, 6))
sns.barplot(x=compras_por_hora.index, y=compras_por_hora.values, palette='mako')
plt.title('Compras por hora del día')
plt.xlabel('Hora')
plt.ylabel('Número de compras (facturas únicas)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# # Analisis de compras por producto y pais

# In[ ]:


producto_mas = pd.crosstab(consumer_data['Dia_semana'], consumer_data['Nombre_Producto'])
producto_mas


# In[ ]:


producto_mas.idxmax(axis=1)


# Se realiza pivot_table

# In[ ]:


producto_mas = consumer_data.pivot_table(index='Nombre_Producto', columns='Dia_semana', aggfunc='size', fill_value=0)
producto_mas.head(6)


# In[ ]:


producto_mas =producto_mas.div(producto_mas.sum(axis=1), axis=0)
producto_mas.head(20)


# In[ ]:





# In[ ]:


# Heatmap de los 10 productos más pedidos por día de la semana (sin sábado), en porcentaje
codigos_excluir = ['POST', 'DOT', 'M', 'BANK CHARGES', 'AMAZONFEE', 'CRUK', 'PADS', 'C2']
# Asegurar que consumer_data y columnas necesarias existan, sin sobrescribir si ya están
if 'consumer_data' not in globals():
    consumer_data = pd.read_excel('data/Online Retail.xlsx')

# Asegurar que Fecha_compra sea datetime
if not pd.api.types.is_datetime64_any_dtype(consumer_data.get('Fecha_compra')):
    consumer_data['Fecha_compra'] = pd.to_datetime(consumer_data['Fecha_compra'], errors='coerce')

# Crear Dia_semana si no existe
if 'Dia_semana' not in consumer_data.columns:
    dias_es = {0:'Lunes',1:'Martes',2:'Miércoles',3:'Jueves',4:'Viernes',5:'Sábado',6:'Domingo'}
    consumer_data['Dia_semana'] = consumer_data['Fecha_compra'].dt.dayofweek.map(dias_es)

# Filtrar códigos a excluir y eliminar sábados (según tu comentario "sin sábado")
datos_productos = consumer_data[~consumer_data['Cod_prod'].isin(codigos_excluir)].copy()
datos_productos = datos_productos[datos_productos['Dia_semana'] != 'Sábado']
top10 = datos_productos['Nombre_Producto'].value_counts().head(10).index
tabla = datos_productos[datos_productos['Nombre_Producto'].isin(top10)]

# % por producto y día (como ya tenías)
pct_producto_dia = pd.crosstab(tabla['Nombre_Producto'], tabla['Dia_semana'], normalize='index') * 100

# Línea base: % global por día, sobre TODA la tienda (sin filtrar a top10)
pct_global_dia = consumer_data['Dia_semana'].value_counts(normalize=True) * 100

# Índice de desviación: >0 significa que ese producto se vende más ese día de lo esperado
desviacion = pct_producto_dia.sub(pct_global_dia, axis=1)

productos_top10 = (
    datos_productos.groupby('Nombre_Producto')['Numero_factura']
    .nunique()
    .sort_values(ascending=False)
    .head(10)
    .index
)

producto_semana_top10 = (
    datos_productos[datos_productos['Nombre_Producto'].isin(productos_top10)]
    .pivot_table(index='Nombre_Producto', columns='Dia_semana', values='Numero_factura', aggfunc='nunique', fill_value=0)
)

orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Domingo']
desviacion = desviacion.reindex(columns=orden_dias)

producto_semana_top10_pct = producto_semana_top10.div(producto_semana_top10.sum(axis=1), axis=0) * 100

plt.figure(figsize=(10, 6))
sns.heatmap(desviacion, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Desviación vs. promedio de la tienda (puntos %)'})
plt.title('Desviación del día de venta de cada producto respecto al promedio general')
plt.xlabel('Día de la semana')
plt.ylabel('Producto')
plt.tight_layout()
plt.show()


# Datos relevantes

# In[ ]:


fecha_corte = consumer_data['Fecha_compra'].max() + pd.Timedelta(days=1)

fecha_corte = consumer_data['Fecha_compra'].max() + pd.Timedelta(days=1)

if 'Monto' not in consumer_data.columns:
    consumer_data['Monto'] = consumer_data['Cantidad'] * consumer_data['Precio unitario']

rfm = consumer_data.groupby('Cod_Cliente').agg(
    Recency=('Fecha_compra', lambda x: (fecha_corte - x.max()).days),
    Frequency=('Numero_factura', 'nunique'),
    Monetary=('Monto', 'sum'),
    Diversidad_productos=('Cod_prod', 'nunique'),
    Pais=('País', 'first')
).reset_index()

rfm['Ticket_promedio'] = rfm['Monetary'] / rfm['Frequency']
if 'Estado de la compra' not in consumer_data.columns:
    consumer_data['Estado de la compra'] = np.where(
        consumer_data['Numero_factura'].astype(str).str.contains('C', case=False, na=False),
        'Cancelado',
        'Realizado'
    )

rfm['Tasa_cancelacion'] = consumer_data.groupby('Cod_Cliente')['Estado de la compra'] \
    .apply(lambda x: (x == 'Cancelado').mean()).values

rfm['Ticket_promedio'] = rfm['Monetary'] / rfm['Frequency']
rfm['Tasa_cancelacion'] = consumer_data.groupby('Cod_Cliente')['Estado de la compra'] \
    .apply(lambda x: (x == 'Cancelado').mean()).values


# In[ ]:


variables = ['Recency', 'Frequency', 'Monetary', 'Diversidad_productos', 'Ticket_promedio', 'Tasa_cancelacion']

fig, axes = plt.subplots(2, 6, figsize=(20, 10))

for i, col in enumerate(variables):
    # Fila 1: distribución cruda
    sns.histplot(rfm[col], kde=True, ax=axes[0, i], color='steelblue')
    axes[0, i].set_title(f'{col} (cruda)')

    # Fila 2: boxplot crudo, para ver outliers de un vistazo
    sns.boxplot(x=rfm[col], ax=axes[1, i], color='salmon')
    axes[1, i].set_title(f'{col} (boxplot)')

plt.tight_layout()
plt.show()


# In[ ]:


def log_signed(x):
    return np.sign(x) * np.log1p(np.abs(x))

rfm_log = rfm.copy()
for col in ['Recency', 'Frequency', 'Diversidad_productos']:
    rfm_log[col] = np.log1p(rfm[col])
for col in ['Monetary', 'Ticket_promedio']:
    rfm_log[col] = log_signed(rfm[col])

fig, axes = plt.subplots(1, 6, figsize=(22, 4))
for i, col in enumerate(variables):
    sns.histplot(rfm_log[col], kde=True, ax=axes[i], color='seagreen')
    axes[i].set_title(f'{col} (log)')
plt.tight_layout()
plt.show()


# In[ ]:


# Gráficos de distribuciones individuales RFM
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Recencia
sns.histplot(rfm['Recency'], kde=True, ax=axes[0], color='steelblue', bins=30)
axes[0].set_title('Distribución de Recencia (días desde última compra)')
axes[0].set_xlabel('Recencia (días)')

# Frecuencia
sns.histplot(rfm['Frequency'], kde=True, ax=axes[1], color='seagreen', bins=30)
axes[1].set_title('Distribución de Frecuencia (número de compras)')
axes[1].set_xlabel('Frecuencia')

# Monetario
sns.histplot(rfm['Monetary'], kde=True, ax=axes[2], color='coral', bins=30)
axes[2].set_title('Distribución de Monetario (monto total gastado)')
axes[2].set_xlabel('Monetario ($)')

plt.tight_layout()
plt.show()


# In[ ]:


# Scatter plots de relaciones RFM
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Recency vs Monetary
axes[0].scatter(rfm['Recency'], rfm['Monetary'], alpha=0.6, s=50, color='steelblue')
axes[0].set_xlabel('Recencia (días)')
axes[0].set_ylabel('Monetario ($)')
axes[0].set_title('Recencia vs Monetario')

# Frecuencia vs Monetary
axes[1].scatter(rfm['Frequency'], rfm['Monetary'], alpha=0.6, s=50, color='seagreen')
axes[1].set_xlabel('Frecuencia (compras)')
axes[1].set_ylabel('Monetario ($)')
axes[1].set_title('Frecuencia vs Monetario')

# Recency vs Frecuencia
axes[2].scatter(rfm['Recency'], rfm['Frequency'], alpha=0.6, s=50, color='coral')
axes[2].set_xlabel('Recencia (días)')
axes[2].set_ylabel('Frecuencia (compras)')
axes[2].set_title('Recencia vs Frecuencia')

plt.tight_layout()
plt.show()


# In[ ]:


# Matriz de correlación RFM
fig, ax = plt.subplots(figsize=(8, 6))
rfm_metricas = rfm[['Recency', 'Frequency', 'Monetary']]
correlacion = rfm_metricas.corr()
sns.heatmap(correlacion, annot=True, cmap='coolwarm', center=0, ax=ax, 
            square=True, linewidths=1, cbar_kws={'label': 'Correlación'})
ax.set_title('Matriz de Correlación de Métricas RFM')
axes[0].set_xlabel('Recencia (días)')
axes[0].set_ylabel('Frecuencia (compras)')
plt.tight_layout()
plt.show()

# Estadísticas descriptivas
print("Estadísticas descriptivas de RFM:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe())


# # Introduciión a Algoritmos no supervisados

# In[ ]:


clientes_df = consumer_data_a.groupby('Cod_Cliente')[['Numero_factura']].nunique()
clientes_df.rename(columns={'Numero_factura': 'num_orders'},inplace=True)
#clientes_df['Dia_semana'] = pd.to_datetime(consumer_data_a['Fecha_compra']).date.dt.day_name()
clientes_df
#consumer_data_a


# In[ ]:


# Una fecha por factura (no por línea de producto)
fechas_por_factura = consumer_data.groupby(['Cod_Cliente', 'Numero_factura'])['Fecha_compra'].min().reset_index()
fechas_por_factura = fechas_por_factura.sort_values(['Cod_Cliente', 'Fecha_compra'])

def promedio_dias(grupo):
    if len(grupo) < 2:
        return np.nan
    return grupo['Fecha_compra'].diff().dropna().dt.days.mean()

dias_entre_compras = fechas_por_factura.groupby('Cod_Cliente').apply(promedio_dias)
rfm['Dias_promedio_entre_compras'] = rfm['Cod_Cliente'].map(dias_entre_compras)

print(f"Clientes con una sola compra (sin intervalo definido): {rfm['Dias_promedio_entre_compras'].isna().sum()} de {len(rfm)}")
clientes_df['num_orders'].hist(bins=30, color='steelblue', edgecolor='black')
clientes_df['avg_time_between_orders'] = rfm['Dias_promedio_entre_compras']
rfm


# In[ ]:


clientes_df['avg_products_per_order'] = consumer_data_a.groupby('Cod_Cliente')['Cod_prod'].count()/clientes_df['num_orders']


# Frcuencia de Compra

# In[ ]:


clientes_df['num_orders'].hist(bins=30, color='darkorange', edgecolor='black')
#clientes_df['Cod_Cliente'] = clientes_df.count()
clientes_df
rfm


# # Clusterización

# In[ ]:


# Importamos las librerías
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(rfm[['Recency','Frequency','Monetary','Ticket_promedio','Tasa_cancelacion']].corr(), annot=True, cmap='coolwarm', center=0, ax=ax, square=True, linewidths=1)
ax.set_title('Matriz de Correlación de Métricas RFM')
plt.tight_layout()
plt.show()


# In[ ]:


def log_signed(x):
    return np.sign(x) * np.log1p(np.abs(x))

cols_finales = ['Recency', 'Frequency', 'Monetary', 'Ticket_promedio', 'Tasa_cancelacion']
rfm_log = rfm.copy()
rfm_log['Recency'] = np.log1p(rfm_log['Recency'])
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary'] = log_signed(rfm_log['Monetary'])
rfm_log['Ticket_promedio'] = log_signed(rfm_log['Ticket_promedio'])
# Tasa_cancelacion ya está en [0,1], no necesita log

scalar = StandardScaler()
scaled_data = scalar.fit_transform(rfm_log[cols_finales])


# In[ ]:


# scaling the data with standardizing the features 
#scalar = StandardScaler()
#scaled_data = scalar.fit_transform(rfm_log[['Recency','Frequency','Monetary','Ticket_promedio','Tasa_cancelacion']]) # ajusta el escalador a los datos y los transforma en un solo paso.


# La separación de atípicos antes de clusterizar: no existe en el notebook

# In[ ]:


from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.012, random_state=42)  # ~54 de 4372 clientes, según lo que vimos antes
rfm_log['Es_atipico'] = iso.fit_predict(scaled_data)  # -1 = atípico, 1 = normal

clientes_atipicos = rfm_log[rfm_log['Es_atipico'] == -1]
rfm_normal = rfm_log[rfm_log['Es_atipico'] == 1].reset_index(drop=True)

print(f"Clientes atípicos separados: {len(clientes_atipicos)}")
print(f"Clientes para segmentación normal: {len(rfm_normal)}")

scaled_normal = scalar.fit_transform(rfm_normal[cols_finales])


# # Estimando un cluster ideal
# 

# Metodo del códo 
# 
# ##### La Métrica Clave: Inercia (Within-Cluster Sum of Squares)
# La métrica más comúnmente utilizada en el método del codo es la inercia, también conocida como Suma de las distancias al cuadrado dentro de los clústeres (WCSS - Within-Cluster Sum of Squares).

# In[ ]:


# implementando el método del codo para determinar el número óptimo de clusters
inertia = []
for i in tqdm(range(1, 11)):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')


# ### El Índice de Calinski-Harabasz para Evaluar Clústeres
# El Índice de Calinski-Harabasz (también conocido como Criterio de Razón de Varianza o VRC) es una métrica utilizada para evaluar la calidad de una solución de clustering. A diferencia del método del codo que se basa en la inspección visual de una curva, este índice proporciona una puntuación numérica que ayuda a determinar el número óptimo de clústeres (k) de una forma más objetiva.
# 
# La idea central es que un buen clustering tiene clústeres muy compactos (baja dispersión dentro de los clústeres) y, al mismo tiempo, muy bien separados entre sí (alta dispersión entre los clústeres). El índice de Calinski-Harabasz cuantifica esta relación.
# 

# In[ ]:


# calculando el índice Calinski-Harabasz para cada número de clusters
CH_scores = []
for i in tqdm(range(2, 11)):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    score = calinski_harabasz_score(scaled_data, kmeans.labels_)
    CH_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), CH_scores, marker='o')
plt.title('Índice Calinski-Harabasz para Determinar el Número Óptimo de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('Índice Calinski-Harabasz')


# 

# In[ ]:


# Implementando KMeans con el número óptimo de clusters
km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = km4.fit_predict(scaled_normal)  # usa el array de clientes sin atipicos
rfm_normal['Cluster'] = kmeans_labels

perfil = rfm_normal.groupby('Cluster')[['Recency','Frequency','Monetary','Ticket_promedio','Tasa_cancelacion']].mean().round(1)
print(perfil)
print(rfm_normal['Cluster'].value_counts())


# In[ ]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Frequency', hue='Cluster', data=rfm_normal,
            palette='coolwarm', legend=True)
plt.yscale('log')
plt.ylabel('Número de órdenes (escala log)')
plt.title('Distribución del número de órdenes por usuario, por clúster')
plt.tight_layout()
plt.show()


# 

# In[ ]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Recency', hue='Cluster', data=rfm_normal,
            palette='coolwarm', legend=True)
#plt.yscale('log')
plt.ylabel('Número de re-órdenes (días desde última compra)')
plt.title('Distribución de frecuencia de compra por usuario, por clúster')
plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Monetary', hue='Cluster', data=rfm_normal,
            palette='coolwarm', legend=True)
#plt.yscale('log')
plt.ylabel('Precio promedio de compra')
plt.title('Distribución de monto de compra por usuario y cancelaciones de pedidos, por clúster')
plt.tight_layout()
plt.show()


# Se analiza en 3d las tres variables

# In[ ]:


import nbformat
print(nbformat.__version__)


# In[5]:


import plotly.graph_objs as go

fig = go.Figure(data=[go.Scatter3d(
    x=rfm_normal.iloc[:,0],
    y=rfm_normal.iloc[:,1],
    z=rfm_normal.iloc[:,2],
    mode='markers',
    marker=dict(
        size=5,
        color=kmeans_labels, # color por cluster
        colorscale='Viridis',
        opacity=0.8
    )
)])

fig.update_layout(
    title='Visualización 3D de los clusters (Plotly)',
    scene=dict(
        xaxis_title="número de órdenes",
        yaxis_title="tiempo promedio entre órdenes",
        zaxis_title="número de productos por orden"
    ),
    width=800,
    height=800,
)

fig.show()


# nbformat
