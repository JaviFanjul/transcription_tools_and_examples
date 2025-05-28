import os
import re
import csv
import unicodedata
from jiwer import wer, mer, cer

# --- Configuración ---

base_path = "./"
folders = {
    "groundtruth": os.path.join(base_path, "groundtruth"),
    "realtime": os.path.join(base_path, "model_large"),
    "batch": os.path.join(base_path, "union"),
}

# Solo archivos .txt para evitar problemas con json u otros
file_names = sorted(f for f in os.listdir(folders["groundtruth"]) if f.lower().endswith(".txt"))

# --- Funciones ---

def normalize_text(text):
    # Elimina acentos
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    # Quita signos de puntuación excepto letras, números y espacios
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Minúsculas y espacios extra
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_texts(text):
    agent_lines = []
    client_lines = []
    full_lines = []

    for line in text.strip().splitlines():
        line = line.strip()
        # Eliminar timestamps si existen
        line = re.sub(r"^\(\d+(\.\d+)?-\d+(\.\d+)?\)\s*", "", line)
        match = re.match(r"^(agente|cliente)\s*:\s*(.*)", line, flags=re.I)
        if match:
            role = match.group(1).lower()
            content = match.group(2).strip()
            content = normalize_text(content)
            if role == "agente":
                agent_lines.append(content)
            elif role == "cliente":
                client_lines.append(content)
            full_lines.append(content)
        else:
            # Si no hay rol explícito, normalizar y añadir igual
            full_lines.append(normalize_text(line))

    agent_text = " ".join(agent_lines)
    client_text = " ".join(client_lines)
    full_text = " ".join(full_lines)

    return full_text, agent_text, client_text

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_metrics(ref, hyp):
    return {
        "wer": round(wer(ref, hyp), 3),
        "mer": round(mer(ref, hyp), 3),
        "cer": round(cer(ref, hyp), 3),
    }

def save_text(output_folder, file_name, text, suffix):
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(file_name)[0]
    path = os.path.join(output_folder, f"{base_name}_{suffix}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# --- Procesamiento ---

results = []

for model in ["batch", "realtime"]:
    for file_name in file_names:
        gt_path = os.path.join(folders["groundtruth"], file_name)
        hyp_path = os.path.join(folders[model], file_name)

        if not os.path.exists(hyp_path):
            continue

        ref_text = read_text(gt_path)
        hyp_text = read_text(hyp_path)

        ref_full, ref_agent, ref_client = extract_texts(ref_text)
        hyp_full, hyp_agent, hyp_client = extract_texts(hyp_text)

        global_metrics = get_metrics(ref_full, hyp_full)
        agent_metrics = get_metrics(ref_agent, hyp_agent)
        client_metrics = get_metrics(ref_client, hyp_client)

        results.append({
            "file": file_name,
            "model": model,
            "wer_global": global_metrics["wer"],
            "mer_global": global_metrics["mer"],
            "cer_global": global_metrics["cer"],
            "wer_agente": agent_metrics["wer"],
            "mer_agente": agent_metrics["mer"],
            "cer_agente": agent_metrics["cer"],
            "wer_cliente": client_metrics["wer"],
            "mer_cliente": client_metrics["mer"],
            "cer_cliente": client_metrics["cer"],
        })

        # Guardar los textos limpios para revisión
        output_dir = os.path.join(base_path, "extracted_texts", model)
        save_text(output_dir, file_name, ref_full, "ref_full")
        save_text(output_dir, file_name, hyp_full, "hyp_full")
        save_text(output_dir, file_name, ref_agent, "ref_agent")
        save_text(output_dir, file_name, hyp_agent, "hyp_agent")
        save_text(output_dir, file_name, ref_client, "ref_client")
        save_text(output_dir, file_name, hyp_client, "hyp_client")

# --- Guardar resultados CSV ---

csv_path = os.path.join(base_path, "results.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Resultados guardados en: {csv_path}")
print(f"✅ Textos extraídos guardados en carpeta: {os.path.join(base_path, 'extracted_texts')}")
