

import json
from tqdm import tqdm
from datasets import load_dataset

from models.model_wrappers import LocalLLM

DATASET_SIZE = 10  # Reduced for testing

dataset = load_dataset(
    "Nan-Do/code-search-net-python",
    split="train"
)

dataset = dataset.shuffle(seed=42).select(range(DATASET_SIZE))

data = [{
    "function_name": x["func_name"],
    "docstring": x["docstring"],
    "code": x["code"]
} for x in dataset]

print(f"Loaded {len(data)} samples")

with open("prompts/code_gen_prompt.txt") as f:
    CODE_GEN_PROMPT = f.read()

model = LocalLLM(
    "Qwen/Qwen2-0.5B-Instruct"
)

from evaluation.memorization_metrics import calculate_minhash_similarity, check_exact_match
from evaluation.robustness import perturb_docstring

results = []

for sample in tqdm(data):
    # 1. Standard Generation
    prompt = CODE_GEN_PROMPT.replace(
        "{FUNCTION_NAME}", sample["function_name"]
    ).replace(
        "{DOCSTRING}", sample["docstring"]
    )
    
    output = model.generate(prompt)
    
    # Metrics
    minhash = calculate_minhash_similarity(output, sample["code"])
    exact = check_exact_match(output, sample["code"])
    
    # 2. Robustness Generation (Perturbed Docstring)
    perturbed_doc = perturb_docstring(sample["docstring"])
    prompt_perturbed = CODE_GEN_PROMPT.replace(
        "{FUNCTION_NAME}", sample["function_name"]
    ).replace(
        "{DOCSTRING}", perturbed_doc
    )
    
    output_perturbed = model.generate(prompt_perturbed)
    minhash_perturbed = calculate_minhash_similarity(output_perturbed, sample["code"])
    
    results.append({
        "function_name": sample["function_name"],
        "original_minhash": minhash,
        "original_exact": exact,
        "perturbed_minhash": minhash_perturbed,
        "robustness_drop": minhash - minhash_perturbed
    })

import numpy as np
avg_sim = np.mean([r['original_minhash'] for r in results])
avg_robust_sim = np.mean([r['perturbed_minhash'] for r in results])
exact_matches = sum([r['original_exact'] for r in results])

print(f"Average MinHash Similarity: {avg_sim:.4f}")
print(f"Average Robustness Similarity: {avg_robust_sim:.4f}")
print(f"Exact Matches: {exact_matches}/{len(results)}")
print(f"Robustness Drop: {avg_sim - avg_robust_sim:.4f}")

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to results.json")

import random

random_hits = sum(random.choice([True, False]) for _ in range(len(data)))
print("Random baseline coverage:", random_hits / len(data))

