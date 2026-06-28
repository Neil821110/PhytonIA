import json
import codecs

with codecs.open('Actividad4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "variables = ['Recency', 'Frequency', 'Monetary', 'Ticket_promedio', 'Tasa_cancelacion']\n",
        "\n",
        "fig, axes = plt.subplots(1, 5, figsize=(20, 5))\n",
        "\n",
        "for i, var in enumerate(variables):\n",
        "    sns.boxplot(x='Cluster', y=var, data=rfm_perfil, ax=axes[i], palette='viridis')\n",
        "    axes[i].set_title(f'Distribución de {var}')\n",
        "    axes[i].set_xlabel('Cluster')\n",
        "    axes[i].set_ylabel(var)\n",
        "    # Aplicar escala logarítmica a variables con amplio rango si es necesario\n",
        "    if var in ['Recency', 'Frequency', 'Monetary', 'Ticket_promedio']:\n",
        "        axes[i].set_yscale('log')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
}

nb['cells'].append(new_cell)

with codecs.open('Actividad4.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
