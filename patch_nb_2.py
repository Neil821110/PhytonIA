import json
import codecs

with codecs.open('Actividad4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'go.Scatter3d' in "".join(cell.get('source', [])):
        # Correct the astype(str) error
        new_source = []
        for line in cell['source']:
            if 'color=kmeans_labels.astype(str)' in line:
                line = line.replace('.astype(str)', '')
            new_source.append(line)
        cell['source'] = new_source

with codecs.open('Actividad4.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
