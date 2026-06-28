import json
import codecs

with codecs.open('Actividad4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "La siguiente tabla resume el tamaño de cada clúster y el promedio de cada variable en k-means\n"
    ]
}

nb['cells'].append(new_cell)

with codecs.open('Actividad4.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
