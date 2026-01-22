# LLM Memorization & Understanding in Software Engineering

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

This project evaluates whether Large Language Models (LLMs) *understand* software engineering tasks or merely *memorize* training data. It adopts methodologies from recent research to distinguish verbatim recall from robust generalization under prompt perturbations.

## 🚀 Getting Started

### Prerequisites

* **Google Colab** (recommended for free GPU access)
* **Python 3.10+** (for local execution)
* **Git**

---

## ☁️ Running on Google Colab (Recommended)

Google Colab provides free access to **T4 GPUs**, which is particularly useful for 7B models such as Mistral or LLaMA.

1. Open a new notebook in Google Colab.
2. Change the runtime type to **T4 GPU**:
   `Runtime` → `Change runtime type` → `T4 GPU`.
3. Run the following commands in a notebook cell to set up the environment:

```python
# 1. Clone the repository
!git clone https://github.com/itsmealessandro/private_ml4se_memorization.git
%cd private_ml4se_memorization

# 2. Install dependencies
!pip install -r requirements.txt

# 3. (Optional) Log in to Hugging Face for gated models (LLaMA / Mistral)
# from huggingface_hub import login
# login()
```

4. Run the experiment:

```python
!python run_experiment.py --n 50 --model "Qwen/Qwen2-0.5B-Instruct"
```

---

## 💻 Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/itsmealessandro/private_ml4se_memorization.git
cd private_ml4se_memorization
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
# .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the full experiment with default settings (Qwen-0.5B by default):

```bash
python run_experiment.py
```

### Common Options

Run with more samples and specify a model:

```bash
python run_experiment.py \
  --n 100 \
  --model "Qwen/Qwen2-0.5B-Instruct" \
  --output results_qwen.json
```

---

## 🔐 Using Restricted Models (Llama, Mistral)

To use gated models such as **Llama-2**, **Llama-3**, or **Mistral**, you must:

1. Have a Hugging Face account
2. Accept the model license on Hugging Face
3. Log in from the terminal:

```bash
huggingface-cli login
```

Then run:

```bash
python run_experiment.py \
  --model "meta-llama/Llama-2-7b-chat-hf" \
  --n 50
```

---

## 📊 Methodology

### 1. Task: Code Generation

The model is given:

* A **function name**
* A **docstring** (from the CodeSearchNet dataset)

The goal is to generate the corresponding Python function implementation.

---

### 2. Metrics

Three complementary metrics are used:

#### 🔹 MinHash Similarity (Jaccard)

Measures structural similarity between generated code and ground truth using token overlap.

* High similarity (> 0.7) → potential memorization

#### 🔹 Exact Match

Checks whether the generated code is reproduced *verbatim*.

#### 🔹 Robustness Drop (Understanding Proxy)

1. Perturb the input docstring (e.g. paraphrasing, adding prefixes like *"Please implement…"*)
2. Generate code again
3. Compare MinHash similarity before and after perturbation

**Interpretation:**

* Small drop → robust behavior → likely *understanding*
* Large drop → brittle behavior → likely *memorization*

---

### 3. Output

The experiment produces an `analysis_<timestamp>/` directory containing:

* **results.json** – raw per-sample results
* **REPORT.md** – human-readable summary, including:

  * Verdict: `STRONG_EVIDENCE_MEMORIZATION` or `LITTLE_EVIDENCE_MEMORIZATION`
  * Aggregate statistics and background similarity baselines
  * Per-sample metrics:

    * `original_minhash`
    * `original_exact`
    * `perturbed_minhash`
    * `robustness_drop`

---

## 🤖 Models & Hardware

We have tested this project with the following models:

| Model                                  | Parameters | Requires Token? | Hardware Requirements |
| -------------------------------------- | ---------- | --------------- | --------------------- |
| **Qwen/Qwen2-0.5B-Instruct (Default)** | 0.5B       | No              | CPU / 8GB RAM         |
| **mistralai/Mistral-7B-Instruct-v0.2** | 7B         | Yes             | T4 GPU / 16GB RAM     |
| **meta-llama/Llama-2-7b-chat-hf**      | 7B         | Yes (Gated)     | T4 GPU / 16GB RAM     |

### ⚠️ Hardware Notes

* **0.5B models**: Run easily on Google Colab (CPU) or standard local laptops.
* **7B models**: Require a T4 GPU (available on Colab Free Tier) or ~16GB RAM locally.

If you encounter **"Killed"** or **OOM** errors, stick to Qwen or consider using quantization (requires code modification).
