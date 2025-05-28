import os
import re
import csv
from sentence_transformers import SentenceTransformer, util

# --- Configuración ---
base_path = "./"
folders = {
    "groundtruth": os.path.join(base_path, "groundtruth"),
    "realtime": os.path.join(base_path, "model_large"),
    "batch": os.path.join(base_path, "union"),
}

file_names = sorted(f for f in os.listdir(folders["groundtruth"]) if f.lower().endswith(".txt"))

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_texts_by_speaker(text):
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

model = SentenceTransformer('all-MiniLM-L6-v2')

results = []

for file_name in file_names:
    gt_path = os.path.join(folders["groundtruth"], file_name)
    realtime_path = os.path.join(folders["realtime"], file_name)
    batch_path = os.path.join(folders["batch"], file_name)

    if not (os.path.exists(realtime_path) and os.path.exists(batch_path)):
        continue

    gt_text = read_text(gt_path)
    rt_text = read_text(realtime_path)
    batch_text = read_text(batch_path)

    gt_full, gt_agent, gt_client = extract_texts_by_speaker(gt_text)
    rt_full, rt_agent, rt_client = extract_texts_by_speaker(rt_text)
    batch_full, batch_agent, batch_client = extract_texts_by_speaker(batch_text)

    gt_emb_full = model.encode(gt_full, convert_to_tensor=True)
    rt_emb_full = model.encode(rt_full, convert_to_tensor=True)
    batch_emb_full = model.encode(batch_full, convert_to_tensor=True)

    gt_emb_agent = model.encode(gt_agent, convert_to_tensor=True)
    rt_emb_agent = model.encode(rt_agent, convert_to_tensor=True)
    batch_emb_agent = model.encode(batch_agent, convert_to_tensor=True)

    gt_emb_client = model.encode(gt_client, convert_to_tensor=True)
    rt_emb_client = model.encode(rt_client, convert_to_tensor=True)
    batch_emb_client = model.encode(batch_client, convert_to_tensor=True)

    sim_rt_full = util.pytorch_cos_sim(gt_emb_full, rt_emb_full).item()
    sim_batch_full = util.pytorch_cos_sim(gt_emb_full, batch_emb_full).item()

    sim_rt_agent = util.pytorch_cos_sim(gt_emb_agent, rt_emb_agent).item()
    sim_batch_agent = util.pytorch_cos_sim(gt_emb_agent, batch_emb_agent).item()

    sim_rt_client = util.pytorch_cos_sim(gt_emb_client, rt_emb_client).item()
    sim_batch_client = util.pytorch_cos_sim(gt_emb_client, batch_emb_client).item()

    results.append({
        "file": file_name,
        "sim_realtime_full": round(sim_rt_full, 4),
        "sim_batch_full": round(sim_batch_full, 4),
        "sim_realtime_agent": round(sim_rt_agent, 4),
        "sim_batch_agent": round(sim_batch_agent, 4),
        "sim_realtime_client": round(sim_rt_client, 4),
        "sim_batch_client": round(sim_batch_client, 4),
    })

csv_path = os.path.join(base_path, "semantic_similarity_results.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "file",
        "sim_realtime_full",
        "sim_batch_full",
        "sim_realtime_agent",
        "sim_batch_agent",
        "sim_realtime_client",
        "sim_batch_client",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Resultados guardados en: {csv_path}")
