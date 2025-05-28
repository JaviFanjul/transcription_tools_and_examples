import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = "semantic_similarity_results.csv"
df = pd.read_csv(csv_path, delimiter=';')

labels = df["file"]
x = np.arange(len(labels))
width = 0.35  # ancho de las barras

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

def plot_similarity(title, col_rt, col_batch, filename):
    fig, ax = plt.subplots(figsize=(12, 5))
    bars_rt = ax.bar(x - width/2, df[col_rt], width, label='Realtime')
    bars_batch = ax.bar(x + width/2, df[col_batch], width, label='Batch')

    ax.set_xlabel('Archivos')
    ax.set_ylabel('Similitud Semántica (Cosine Similarity)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()

    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath)
    print(f"✅ Gráfico guardado en: {filepath}")

    plt.show()
    plt.close(fig)  # Cerrar figura para que no interfiera con la siguiente

plot_similarity('Similitud Semántica - Texto Completo', 'sim_realtime_full', 'sim_batch_full', 'similarity_full.png')
plot_similarity('Similitud Semántica - Agente', 'sim_realtime_agent', 'sim_batch_agent', 'similarity_agent.png')
plot_similarity('Similitud Semántica - Cliente', 'sim_realtime_client', 'sim_batch_client', 'similarity_client.png')
