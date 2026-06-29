import json
import codecs

with codecs.open('Actividad4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and '# 3. Calcular el perfil con valores reales' in ''.join(cell.get('source', [])):
        cell['source'] = [
            "# 1. Entrenar el K-means y obtener las etiquetas\n",
            "km4 = KMeans(n_clusters=4, random_state=42, n_init='auto')\n",
            "kmeans_labels_rfm = km4.fit_predict(scaled_normal)\n",
            "rfm_normal['Cluster'] = kmeans_labels_rfm\n",
            "\n",
            "# 2. Traer los valores crudos (sin log) usando Cod_Cliente como llave\n",
            "rfm_crudo = rfm.set_index('Cod_Cliente')\n",
            "rfm_perfil = rfm_crudo.loc[rfm_normal['Cod_Cliente']].reset_index()\n",
            "rfm_perfil['Cluster'] = kmeans_labels_rfm\n",
            "\n",
            "# 3. Calcular el perfil con valores reales en un solo reporte\n",
            "columnas = ['Recency', 'Frequency', 'Monetary', 'Ticket_promedio', 'Tasa_cancelacion']\n",
            "perfil = rfm_perfil.groupby('Cluster')[columnas].mean().round(1)\n",
            "\n",
            "# Adicionar la columna de número de clientes al inicio para el reporte\n",
            "perfil.insert(0, 'No. Clientes', rfm_perfil['Cluster'].value_counts())\n",
            "\n",
            "display(perfil)\n"
        ]
        break

with codecs.open('Actividad4.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
