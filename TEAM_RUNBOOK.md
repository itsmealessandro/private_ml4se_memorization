# Team runbook (3 persone) — Project 8: Memorization of public SE datasets in LLMs

Questo file serve a dividerci il lavoro: ognuno runna **solo il nostro codice** (`run_experiment.py`) su **modelli diversi** (quelli indicati dalla consegna) e produce output (JSON) comparabili per scrivere il report finale.

## Deliverable minimo per il report
Per ogni run vogliamo avere (almeno):
- modello, dataset, split, n, max_new_tokens, seed
- metriche aggregate: `avg_original_minhash`, `exact_match_rate`, `high_similarity_rate@0.7`, `avg_robustness_drop`
- (opzionale) esempi qualitativi: 2–3 casi “molto simili” e 2–3 casi “molto diversi”

Nota: al momento la pipeline è già pronta per **CodeSearchNet (Python)** via HuggingFace. Se aggiungiamo altri dataset (StackOverflow/CWE/CodeXGlue) useremo lo stesso schema di run, ma per ora dividiamoci il lavoro sui **modelli**.

---

## Persona A — Qwen (consigliato) su CodeSearchNet
**Obiettivo:** risultati di riferimento su un modello piccolo ma code-capable (Qwen) con run “da report”.

**Cosa runnare (GPU consigliata):**
```bash
cd /home/omalex/projects/llm_memorization_project

mkdir -p runs

# run piccola per controllare che tutto regga
./.venv/bin/python run_experiment.py \
  --dataset Nan-Do/code-search-net-python \
  --split train \
  --n 20 \
  --model Qwen/Qwen2-0.5B-Instruct \
  --max-new-tokens 128 \
  --output runs/qwen2_0.5b_csn_train_n20_t128_seed42.json

# run da report (se non va in OOM)
./.venv/bin/python run_experiment.py \
  --dataset Nan-Do/code-search-net-python \
  --split train \
  --n 50 \
  --model Qwen/Qwen2-0.5B-Instruct \
  --max-new-tokens 256 \
  --output runs/qwen2_0.5b_csn_train_n50_t256_seed42.json

# controllo: senza perturbazione (ablation)
./.venv/bin/python run_experiment.py \
  --dataset Nan-Do/code-search-net-python \
  --split train \
  --n 50 \
  --model Qwen/Qwen2-0.5B-Instruct \
  --max-new-tokens 256 \
  --no-perturb \
  --output runs/qwen2_0.5b_csn_train_n50_t256_seed42_no_perturb.json
```

**Cosa consegnare al gruppo:**
- i file `runs/qwen2_0.5b_*.json`
- 3 righe di commento: (1) `high_similarity_rate@0.7`/`exact_match_rate`, (2) `avg_robustness_drop`, (3) tempo/OOM.

---

## Persona B — Llama (consigliato) su CodeSearchNet
**Obiettivo:** replicare gli stessi test del Ruolo A ma su una variante Llama accessibile.

Modelli candidati (scegline uno che il tuo PC regge):
- `meta-llama/Llama-3.2-1B-Instruct` (potrebbe richiedere accesso HF e più VRAM; se OOM, riduci token/n)
- (fallback leggero, se Llama gated/non disponibile) `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

**Cosa runnare:**
```bash
cd /home/omalex/projects/llm_memorization_project

mkdir -p runs

# scegli MODEL in base a cosa riesci a scaricare/hostare
MODEL=meta-llama/Llama-3.2-1B-Instruct

./.venv/bin/python run_experiment.py \
  --dataset Nan-Do/code-search-net-python \
  --split train \
  --n 20 \
  --seed 42 \
  --model "$MODEL" \
  --max-new-tokens 128 \
  --output runs/llama_csn_train_n20_t128_seed42.json

./.venv/bin/python run_experiment.py \
  --dataset Nan-Do/code-search-net-python \
  --split train \
  --n 50 \
  --seed 42 \
  --model "$MODEL" \
  --max-new-tokens 256 \
  --output runs/llama_csn_train_n50_t256_seed42.json
```

**Cosa consegnare al gruppo:**
- i file `runs/llama_*.json` (o `runs/tinyllama_*.json` se fallback)
- 3 righe di commento come per Persona A.

---

## Persona C — Mistral / gpt-oss-20B (consigliati) su CodeSearchNet
**Obiettivo:** coprire un terzo modello “della consegna” e confrontare con Qwen/Llama.

Scelta pratica (in base alle risorse):
- **Mistral**: se hai una GPU più grande o puoi usare CPU (molto lento). Se non regge, documenta il motivo (VRAM) e usa un fallback leggero.
- **gpt-oss-20B**: di solito richiede endpoint/serving dedicato; se non disponibile localmente, documenta la limitazione e usa un modello open più piccolo.

**Run (template):**
```bash
cd /home/omalex/projects/llm_memorization_project

mkdir -p runs

# Esempio: sostituisci con un modello Mistral che riesci a far girare
MODEL=mistralai/Mistral-7B-Instruct-v0.2

./.venv/bin/python run_experiment.py \
  --dataset Nan-Do/code-search-net-python \
  --split train \
  --n 20 \
  --seed 42 \
  --model "$MODEL" \
  --max-new-tokens 128 \
  --output runs/mistral_csn_train_n20_t128_seed42.json
```

**Cosa consegnare al gruppo:**
- i file `runs/mistral_*.json` (o equivalente)
- nota chiara su fattibilità (VRAM/tempo) e, se serve, fallback usato.

---

## Come leggere i risultati (per il report)
Ogni file prodotto da `run_experiment.py` contiene:
- `summary`: numeri aggregati da mettere in tabella
- `results`: per-esempio

Interpretazione (proxy, non prova definitiva di training-leak):
- `exact_match_rate` o `high_similarity_rate@0.7` > 0 → segnali compatibili con recall/copia.
- `avg_robustness_drop` alto e consistente → segnali compatibili con comportamento “trigger-based”.
- valori bassi + drop vicino a 0 → più compatibile con assenza di memorization (o modello che non sa fare bene il task).

---

## Troubleshooting rapido
- OOM/VRAM: riduci `--max-new-tokens` (128), riduci `--n` (20–30), oppure usa `--no-perturb`.
- Run troppo lenta: stessa soluzione; su CPU i tempi crescono molto.
- Output “sporco” (testo extra): il parsing è in `evaluation/output_parsing.py`.

