# 🕵️‍♂️ Project 8: Memorization of public SE datasets in LLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**Course:** Machine Learning for Software Engineering (DT1052)  
**University:** University of L'Aquila  
**Academic Year:** 2025/2026

## 👥 Team Members
* **Omar Dinari** 
* **Alessandro Di Giacomo** 
* **Agostino D'Agostino** 

---

## 📖 Overview
Large Language Models (LLMs) are powerful code assistants, but do they actually *understand* code, or do they just *memorize* their training data? 

This project investigates **Memorization vs. Generalization** in three state-of-the-art models using the **CodeSearchNet** dataset:
1. **Qwen2-0.5B-Instruct**
2. **Llama-2-7b-chat**
3. **Mistral-7B-Instruct-v0.2**

We employ a probing framework based on **MinHash Similarity**, **Exact Match**, and **Docstring Perturbation** to quantify data leakage and robustness.

---

## 🚀 Key Features
* **Modular Architecture:** Separation of concerns between model wrappers, evaluation metrics, and experiment logic.
* **Data Persistence Flexibility:** Supports both **In-Memory** loading (fast) and **Streaming** (RAM-efficient) via the `--streaming` flag.
* **Sensitivity Analysis:** Evaluates model stability across varying sample sizes ($N=5, 50, 100$) to filter out background noise.
* **Robustness Metrics:** Calculates $\Delta R$ (Robustness Drop) to distinguish between rote memorization and semantic understanding.

---

## 📂 Project Structure

```text
├── models/                  # LLM Wrappers (Hugging Face integration)
│   └── model_wrappers.py
├── evaluation/              # Metrics (MinHash, Exact Match) & Robustness logic
│   ├── memorization_metrics.py
│   └── robustness.py
├── prompts/                 # Prompt templates
│   └── code_gen_prompt.txt
├── report/                  # 📄 FINAL REPORT & LATEX SOURCE
│   ├── main.tex
│   └── report.pdf
├── run_experiment.py        # Main entry point for experiments
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## ☁️ Running on Google Colab (Recommended)

Google Colab provides free access to **T4 GPUs**, which is required for 7B models like Mistral.

### 1. Clone the repository
```python
!git clone https://github.com/itsmealessandro/private_ml4se_memorization.git
%cd private_ml4se_memorization
```

### 2. Install dependencies
```python
!pip install -r requirements.txt
```

### 3. Login to Hugging Face (Required for Llama/Mistral)
```python
from huggingface_hub import login
login()
```

---

## 💻 Usage

### 1. Standard Run (In-Memory)
```bash
python run_experiment.py --model "Qwen/Qwen2-0.5B-Instruct" --n 50 --output "results_qwen.json"
```

### 2. Streaming Run (Low RAM)
```bash
python run_experiment.py --model "mistralai/Mistral-7B-Instruct-v0.2" --n 100 --streaming
```

---

## 📊 Summary of Results

| Model | Avg Similarity | Exact Match Rate | Interpretation |
| --- | --- | --- | --- |
| **Qwen-0.5B** | 0.0602 | **0.00%** | Weak structure, no memorization. |
| **Llama-2-7B** | 0.0923 | **0.00%** | Moderate structure, no memorization. |
| **Mistral-7B** | **0.1508** | **0.00%** | **Strongest Generalization**, zero leakage. |

> All models maintained a **0.00% Exact Match rate**, confirming robust privacy preservation and lack of verbatim regurgitation.

---

## 🔍 Methodology Details

### Metrics
* **MinHash Similarity**
* **Exact Match**
* **Robustness Drop ($\Delta R$)**

### Hardware Requirements

| Model | Hardware | Note |
| --- | --- | --- |
| **Qwen-0.5B** | CPU / 8GB RAM | Runs on standard laptops. |
| **Mistral-7B** | T4 GPU / 16GB RAM | Requires GPU+HF Token. |
| **Llama-2-7B** | T4 GPU / 16GB RAM | Requires GPU + HF Token. |
