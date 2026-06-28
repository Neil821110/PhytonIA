import pandas as pd
import numpy as np
import plotly.graph_objs as go

# We don't have the live variables, let's just create dummy ones that mimic them
rfm_normal = pd.DataFrame({
    'Cod_Cliente': [1, 2, 3],
    'Dias_promedio_entre_compras': [10.5, 20.1, 5.0]
})
rfm = pd.DataFrame({
    'Cod_Cliente': [1, 2, 3],
    'Frequency': [10, 20, 30],
    'Diversidad_productos': [2, 4, 6]
})
kmeans_labels = np.array([0, 1, 0])

try:
    rfm_crudo = rfm.set_index('Cod_Cliente')

    fig = go.Figure(data=[go.Scatter3d(
        x=rfm_normal['Cod_Cliente'].map(rfm_crudo['Frequency']),
        y=rfm_normal['Dias_promedio_entre_compras'],
        z=rfm_normal['Cod_Cliente'].map(rfm_crudo['Diversidad_productos']),
        mode='markers',
        marker=dict(
            size=5,
            color=kmeans_labels.astype(str),
            colorscale='Viridis',
            opacity=0.6
        )
    )])
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
