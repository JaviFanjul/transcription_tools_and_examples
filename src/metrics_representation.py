import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CARGA DEL CSV ===
df = pd.read_csv("./results.csv", delimiter=";")

# Crear carpeta para guardar los gráficos
os.makedirs("graficos", exist_ok=True)

# Métricas a graficar
metricas = ["wer", "mer", "cer"]
tipos = ["global", "agente", "cliente"]

# === BOXPLOTS POR MÉTRICA Y TIPO ===
for metrica in metricas:
    for tipo in tipos:
        col = f"{metrica}_{tipo}"
        if col not in df.columns:
            continue

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="model", y=col)
        plt.title(f"{metrica.upper()} ({tipo}) por modelo")
        plt.tight_layout()
        plt.savefig(f"graficos/boxplot_{col}.png")
        plt.close()

# === BARRAS AGRUPADAS POR ARCHIVO Y TIPO ===
for metrica in metricas:
    for tipo in tipos:
        col = f"{metrica}_{tipo}"
        if col not in df.columns:
            continue

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="file", y=col, hue="model")
        plt.title(f"{metrica.upper()} ({tipo}) por archivo y modelo")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"graficos/barras_{col}.png")
        plt.close()
