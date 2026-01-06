# LLM Memorization & Understanding in Software Engineering

This project evaluates whether Large Language Models (LLMs) "understand" software engineering tasks or simply "memorize" training data. It implements methodologies from recent research (Paper 2502 & 2505) to distinguish between the two.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A virtual environment is recommended.

### Installation
1.  **Clone the repository** (if applicable).
2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🧪 Running the Experiment

The core experiment is defined in `main.ipynb`. You can run it using Jupyter Notebook or the provided script.

### Option 1: Jupyter Notebook
1.  Start Jupyter:
    ```bash
    jupyter notebook
    ```
2.  Open `main.ipynb`.
3.  Run all cells.

### Option 2: Python Script
We provide a helper script to run the experiment directly from the terminal:
```bash
python3 run_experiment.py
```

## 📊 Methodology

### 1. Task: Code Generation
The model is given a **Function Name** and a **Docstring** and must generate the corresponding Python code.

### 2. Metrics
We use three key metrics to evaluate the model:

*   **MinHash Similarity (Jaccard)**: Measures the structural similarity between the generated code and the ground truth. High similarity (>0.7) suggests potential memorization.
*   **Exact Match**: Checks if the model reproduces the code verbatim.
*   **Robustness Drop (Understanding Proxy)**:
    *   We **perturb** the input docstring (e.g., paraphrasing, adding prefixes).
    *   We compare the MinHash similarity of the output from the *original* docstring vs. the *perturbed* docstring.
    *   **Interpretation**:
        *   **Small Drop**: The model is robust -> Likely **Understanding**.
        *   **Large Drop**: The model fails when the prompt changes slightly -> Likely **Memorization** (trigger-based).

## 🤖 Models
The project is currently configured to use **Qwen/Qwen2-0.5B-Instruct** for fast iteration. You can change the model in `main.ipynb` by modifying the `LocalLLM` initialization.

## 📂 Project Structure
- `data/`: Stores downloaded datasets (cached by Hugging Face).
- `evaluation/`: Contains metric implementations.
    - `memorization_metrics.py`: MinHash and Exact Match logic.
    - `robustness.py`: Docstring perturbation logic.
- `models/`: Contains `model_wrappers.py` for LLM inference.
- `prompts/`: Contains prompt templates.
