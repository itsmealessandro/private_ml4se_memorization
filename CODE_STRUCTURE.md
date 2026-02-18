# 📁 Code Structure - Complete Documentation

## Project Overview

This project analyzes **Memorization vs. Generalization** in Large Language Models (LLMs) using the CodeSearchNet dataset. The goal is to verify whether models memorize code verbatim from training data or truly generalize.

---

## 🏗️ General Architecture

The project follows a modular architecture with separation of concerns:

```
private_ml4se_memorization/
│
├── run_experiment.py          # ⭐ Main script - entry point
│
├── models/                     # 🤖 LLM wrappers
│   └── model_wrappers.py
│
├── evaluation/                 # 📊 Metrics and evaluation
│   ├── memorization_metrics.py  # MinHash similarity & exact match
│   ├── output_parsing.py        # Code extraction from responses
│   ├── robustness.py            # Docstring perturbation
│   └── metrics.py               # Support file
│
├── prompts/                    # 💬 Prompt templates
│   ├── code_gen_prompt.txt
│   └── function_prompt.txt
│
├── reports/                    # 📄 Generated reports (timestamped)
│   └── analysis_YYYYMMDD_HHMM/
│       ├── results.json
│       └── REPORT.md
│
├── analysis_results/           # 📈 Analysis results (deprecated/legacy)
├── data/                       # 💾 Dataset placeholder (empty)
├── Documentation/              # 📚 LaTeX report and PDF
│   ├── main.tex
│   └── LLM_Memorization_vs_Understanding.pdf
│
├── requirements.txt            # 📦 Python dependencies
├── README.md                   # 📖 User documentation
└── MAIN_PAPER.pdf             # 📄 Reference paper
```

---

## 🔍 Main Components

### 1️⃣ `run_experiment.py` (462 lines)

**Role**: Main script that orchestrates the entire experiment.

#### Execution Flow:

1. **Dataset Loading**
   - Supports two modes:
     - **In-Memory** (default): loads entire dataset into RAM
     - **Streaming** (`--streaming` flag): loads data on-the-fly to save RAM
   - Default dataset: `Nan-Do/code-search-net-python` (CodeSearchNet)

2. **Model Preparation**
   - Loads specified LLM model (default: Qwen2-0.5B-Instruct)
   - Supports: Qwen, Llama-2, Mistral

3. **Probing Loop** (for each example):
   - Formats prompt with function name + docstring
   - **Original generation**: asks model to generate code
   - Extracts code from response
   - Calculates:
     - **MinHash similarity**: similarity between generated and reference code
     - **Exact match**: checks if code is identical
   - (Optional) **Perturbed generation**:
     - Perturbs the docstring
     - Regenerates code
     - Calculates **robustness drop** = original_similarity - perturbed_similarity

4. **Global Metrics Calculation**
   - Average similarity
   - Exact match rate
   - Background similarity (baseline with random pairs)
   - Average robustness drop

5. **Report Generation**
   - `results.json`: machine-readable data
   - `REPORT.md`: human-readable report with verdict:
     - **STRONG_EVIDENCE_MEMORIZATION**: high similarity + fragility to perturbations
     - **LITTLE_EVIDENCE_MEMORIZATION**: low similarity close to background
     - **INCONCLUSIVE**: mixed signals

#### Key Functions:

```python
def run(args) -> dict:
    """Executes complete experiment and returns report."""
    
def _verdict_from_summary(summary, thresholds) -> (str, str):
    """Determines verdict based on metrics."""
    
def _render_human_report(report, thresholds, results_path) -> str:
    """Generates human-readable Markdown report."""
    
def _background_similarity(codes, seed) -> float:
    """Calculates baseline: similarity between random code pairs."""
```

#### CLI Parameters:

```bash
--model          # Hugging Face model (default: Qwen2-0.5B-Instruct)
--dataset        # Dataset (default: CodeSearchNet Python)
--n              # Number of samples (default: 10)
--seed           # Random seed (default: 42)
--streaming      # Use streaming mode (low RAM)
--perturb        # Enable robustness test (default: True)
--max-new-tokens # Generation length (default: 256)
--output         # Output filename (default: results.json)
```

---

### 2️⃣ `models/model_wrappers.py` (34 lines)

**Role**: Wrapper for loading and using Hugging Face LLM models.

#### `LocalLLM` Class:

```python
class LocalLLM:
    def __init__(self, model_name):
        """Loads tokenizer and model from Hugging Face.
        
        Features:
        - Auto-detect GPU/CPU
        - Float16 on GPU, Float32 on CPU
        - device_map="auto" for automatic distribution
        """
    
    def generate(self, prompt, max_tokens=256, return_full_text=False):
        """Generates text from prompt.
        
        Args:
            prompt: input text
            max_tokens: max new tokens to generate
            return_full_text: if True, returns prompt+completion
        
        Returns:
            Completion (without prompt if return_full_text=False)
        """
```

**Design Pattern**: Facade Pattern - simplifies interaction with Transformers.

---

### 3️⃣ `evaluation/memorization_metrics.py` (30 lines)

**Role**: Implements memorization metrics.

#### Functions:

```python
def calculate_minhash_similarity(text1, text2) -> float:
    """Calculates Jaccard similarity with Theta Sketch (MinHash).
    
    Algorithm:
    1. Tokenize by whitespace
    2. Create sketch for each text
    3. Calculate Jaccard similarity
    
    Returns: similarity in [0, 1] (1 = identical)
    """

def check_exact_match(text1, text2) -> bool:
    """Checks if two texts are identical (ignores whitespace).
    
    Returns: True if exact match
    """
```

**Library used**: `datasketches` for efficient MinHash.

**Why MinHash?**
- Efficient for large texts
- Robust to small variations
- Standard for similarity detection

---

### 4️⃣ `evaluation/output_parsing.py` (27 lines)

**Role**: Extracts Python code from LLM responses.

#### Main Function:

```python
def extract_code(text: str) -> str:
    """Extracts Python code from raw model text.
    
    Strategies (in order):
    1. Search for code fence (```python ... ``` or ``` ... ```)
    2. Fallback: search for first 'def ' or 'class '
    3. Last fallback: return all text
    
    Returns: extracted code (best-effort)
    """
```

**Regex used**: `r"```(?:python)?\s*(.*?)```"`
- Handles both ` ```python` and ` ``` `
- DOTALL flag: multi-line match
- IGNORECASE: case-insensitive

**Problem solved**: Models often add text explanations before/after code.

---

### 5️⃣ `evaluation/robustness.py` (26 lines)

**Role**: Perturbs docstrings to test model robustness.

#### Function:

```python
def perturb_docstring(docstring: str) -> str:
    """Perturbs docstring for robustness test.
    
    Strategies:
    1. Add random prefix:
       - "Implement this function: "
       - "Write a python function that "
       - "Coding task: "
       - "Please help me with this: "
    
    2. Randomly uppercase some words (10% probability)
    
    Returns: perturbed docstring
    """
```

**Rationale**: If the model memorized, a small perturbation should cause similarity to drop. If it generalizes, similarity remains stable.

---

### 6️⃣ `prompts/code_gen_prompt.txt`

**Prompt Template**:

```text
You are an expert Python programmer.
Implement the function described below.
Return only the code.

Function Name: {FUNCTION_NAME}
Docstring: {DOCSTRING}

Code:
```

**Placeholders**:
- `{FUNCTION_NAME}`: replaced with function name
- `{DOCSTRING}`: replaced with docstring

**Design**: Minimal template to avoid bias.

---

## 🔄 Complete Workflow

```
[Start] 
  ↓
[Load Dataset]
  ↓
[Load LLM Model]
  ↓
[For each example...]
  ↓
[Format Prompt]
  ↓
[Generate Original Code]
  ↓
[Extract Code]
  ↓
[Calculate MinHash & Exact Match]
  ↓
[Perturb?] ──Yes──> [Perturb Docstring] → [Generate Perturbed Code] → [Calculate Robustness Drop]
  ↓ No
[Save Results]
  ↓
[More examples?] ──Yes──> [Back to "For each example..."]
  ↓ No
[Calculate Aggregate Metrics]
  ↓
[Calculate Background Baseline]
  ↓
[Determine Verdict]
  ↓
[Generate JSON + MD Report]
  ↓
[End]
```

---

## 📊 Metrics Explained

### MinHash Similarity
- **Range**: [0, 1]
- **0.0**: completely different texts
- **1.0**: identical texts
- **Interpretation**:
  - `< 0.1`: low similarity (generalization)
  - `0.1 - 0.3`: moderate similarity
  - `> 0.7`: high similarity (possible memorization)

### Exact Match Rate
- **Meaning**: % of generations identical verbatim to reference
- **Threshold**: even 1% is suspicious

### Robustness Drop (ΔR)
- **Formula**: `ΔR = similarity_original - similarity_perturbed`
- **Interpretation**:
  - `ΔR ≈ 0`: robust model (generalizes)
  - `ΔR > 0.1`: fragile model (memorizes)

### Background Similarity
- **Calculation**: average similarity between random code pairs
- **Use**: baseline to understand if observed similarity is significant
- **Typically**: 0.03-0.05 for random code

---

## 🎯 Decision Thresholds

```python
T_EXACT_STRONG = 0.05      # 5% exact match → MEMORIZATION
T_MARGIN_STRONG = 0.20     # +0.20 vs background → MEMORIZATION
T_HIGH_SIM_STRONG = 0.20   # 20% samples with sim > 0.7 → MEMORIZATION
T_DROP_STRONG = 0.10       # ΔR > 0.10 → MEMORIZATION

T_EXACT_LOW = 0.01         # < 1% exact match → GENERALIZATION
T_MARGIN_LOW = 0.10        # < +0.10 vs background → GENERALIZATION
T_DROP_LOW = 0.05          # ΔR < 0.05 → GENERALIZATION
```

---

## 🚀 Usage Examples

### Complete Run (Qwen, 50 samples)
```bash
python run_experiment.py \
  --model "Qwen/Qwen2-0.5B-Instruct" \
  --n 50 \
  --output "results_qwen.json"
```

### Streaming Run (Mistral, low RAM)
```bash
python run_experiment.py \
  --model "mistralai/Mistral-7B-Instruct-v0.2" \
  --n 100 \
  --streaming \
  --output "results_mistral.json"
```

### Disable Perturbation
```bash
python run_experiment.py \
  --model "meta-llama/Llama-2-7b-chat-hf" \
  --n 50 \
  --no-perturb
```

---

## 📈 Generated Outputs

### `reports/analysis_YYYYMMDD_HHMM/results.json`

Structure:
```json
{
  "meta": {
    "timestamp": "2026-02-18T10:30:00Z",
    "model": "Qwen/Qwen2-0.5B-Instruct",
    "dataset": "Nan-Do/code-search-net-python",
    "n": 50,
    "seed": 42,
    "perturb": true
  },
  "summary": {
    "n": 50,
    "avg_original_minhash": 0.0602,
    "avg_perturbed_minhash": 0.0589,
    "exact_matches": 0,
    "exact_match_rate": 0.0,
    "avg_robustness_drop": 0.0013,
    "background_ref_minhash": 0.0384,
    "background_gen_minhash": 0.0291,
    "high_similarity_rate@0.7": 0.0
  },
  "results": [
    {
      "function_name": "my_function",
      "original_minhash": 0.0623,
      "original_exact": false,
      "perturbed_minhash": 0.0601,
      "robustness_drop": 0.0022
    },
    ...
  ]
}
```

### `reports/analysis_YYYYMMDD_HHMM/REPORT.md`

Human-readable report with:
- Run info
- Verdict (STRONG_EVIDENCE_MEMORIZATION / LITTLE_EVIDENCE_MEMORIZATION / INCONCLUSIVE)
- Key metrics
- Interpretation

---

## 🔧 Technical Requirements

### Dependencies (`requirements.txt`):
```
torch                # Deep learning framework
transformers         # Hugging Face models
datasets             # Dataset loading
accelerate           # Distributed inference
tqdm                 # Progress bars
datasketches         # MinHash implementation
numpy                # Numerical computing
jupyter              # Notebook support
ipykernel            # Kernel for Jupyter
```

### Hardware:
- **Qwen-0.5B**: CPU / 8GB RAM
- **Llama-2-7B**: GPU (T4) / 16GB RAM + HF Token
- **Mistral-7B**: GPU (T4) / 16GB RAM + HF Token

---

## 🧪 Design Patterns Used

1. **Facade Pattern** (`LocalLLM`): simplifies interface with Transformers
2. **Template Method** (`_format_prompt`): template pattern for prompts
3. **Strategy Pattern** (`perturb_docstring`): different perturbation strategies
4. **Pipeline Pattern** (`run`): sequential orchestration of steps

---

## 🔐 Security and Privacy

- **No API calls**: everything local
- **No telemetry**: no external data sending
- **Fixed seed**: reproducibility
- **Hugging Face Token**: only needed for gated models (Llama, Mistral)

---

## 📚 References

- **Paper**: `MAIN_PAPER.pdf`
- **LaTeX Report**: `Documentation/main.tex`
- **PDF Report**: `Documentation/LLM_Memorization_vs_Understanding.pdf`

---

## 🎓 Academic Context

- **Course**: Machine Learning for Software Engineering (DT1052)
- **University**: University of L'Aquila
- **Academic Year**: 2025/2026
- **Team**: Omar Dinari, Alessandro Di Giacomo, Agostino D'Agostino

---

## 💡 Key Findings

### Results Obtained:
| Model | Avg Similarity | Exact Match | Interpretation |
|---------|---------------|-------------|-----------------|
| Qwen-0.5B | 0.0602 | 0.00% | No memorization |
| Llama-2-7B | 0.0923 | 0.00% | No memorization |
| Mistral-7B | 0.1508 | 0.00% | Best generalization |

**All models** showed:
- Zero exact match (no verbatim reproduction)
- Similarity close to background
- Robustness to perturbations

**Conclusion**: The tested models **generalize** rather than memorize CodeSearchNet.

---

## 🛠️ Maintenance and Extensibility

### To add a new model:
1. Ensure it's compatible with Hugging Face Transformers
2. Use: `python run_experiment.py --model "org/model-name"`

### To add a new metric:
1. Add function in `evaluation/memorization_metrics.py`
2. Modify `run()` in `run_experiment.py` to calculate it
3. Add to `summary` dict

### To add a new dataset:
1. Ensure it has fields `func_name`, `docstring`, `code`
2. Use: `python run_experiment.py --dataset "org/dataset-name"`

---

**Documentation Author**: GitHub Copilot  
**Date**: 2026-02-18  
**Version**: 1.0
