import json
import codecs

with codecs.open('Actividad4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][-2]
cell['source'] = [
    "import plotly.graph_objs as go\n",
    "import numpy as np\n",
    "\n",
    "fig = go.Figure(data=[go.Scatter3d(\n",
    "    x=np.expm1(rfm_normal['Frequency']), # Número de órdenes original\n",
    "    y=rfm_normal['Dias_promedio_entre_compras'], # Tiempo promedio entre órdenes\n",
    "    z=np.expm1(rfm_normal['Diversidad_productos']), # Diversidad de productos\n",
    "    mode='markers',\n",
    "    marker=dict(\n",
    "        size=5,\n",
    "        color=kmeans_labels, # color por cluster\n",
    "        colorscale='Viridis',\n",
    "        opacity=0.8\n",
    "    )\n",
    ")])\n",
    "\n",
    "fig.update_layout(\n",
    "    title='Visualización 3D de los clusters (Plotly)',\n",
    "    scene=dict(\n",
    "        xaxis_title=\"Número de órdenes\",\n",
    "        yaxis_title=\"Tiempo promedio entre órdenes\",\n",
    "        zaxis_title=\"Diversidad de productos\"\n",
    "    ),\n",
    "    width=800,\n",
    "    height=800,\n",
    ")\n",
    "\n",
    "fig.show()"
]

with codecs.open('Actividad4.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
