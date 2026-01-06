import argparse
import json
import random
from datetime import datetime

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from evaluation.memorization_metrics import calculate_minhash_similarity, check_exact_match
from evaluation.output_parsing import extract_code
from evaluation.robustness import perturb_docstring
from models.model_wrappers import LocalLLM


DEFAULT_DATASET = "Nan-Do/code-search-net-python"
DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"


def _load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _format_prompt(template: str, function_name: str, docstring: str) -> str:
    return (
        template.replace("{FUNCTION_NAME}", function_name or "")
        .replace("{DOCSTRING}", docstring or "")
    )


def _background_similarity(codes: list[str], seed: int) -> float:
    if len(codes) < 2:
        return 0.0
    rng = random.Random(seed)
    idxs = list(range(len(codes)))
    rng.shuffle(idxs)
    shuffled = [codes[i] for i in idxs]
    sims = [
        calculate_minhash_similarity(a, b)
        for a, b in zip(codes, shuffled)
        if a and b
    ]
    return float(np.mean(sims)) if sims else 0.0


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)

    prompt_template = _load_prompt(args.prompt)

    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.shuffle(seed=args.seed).select(range(args.n))

    samples = [
        {
            "function_name": x.get("func_name", ""),
            "docstring": x.get("docstring", ""),
            "code": x.get("code", ""),
        }
        for x in dataset
    ]

    model = LocalLLM(args.model)

    results: list[dict] = []
    generated_codes: list[str] = []
    reference_codes: list[str] = []

    for i, sample in enumerate(tqdm(samples, desc="Probing")):
        prompt = _format_prompt(prompt_template, sample["function_name"], sample["docstring"])
        completion = model.generate(prompt, max_tokens=args.max_new_tokens)
        gen_code = extract_code(completion)

        ref_code = sample["code"] or ""
        minhash = calculate_minhash_similarity(gen_code, ref_code)
        exact = check_exact_match(gen_code, ref_code)

        minhash_perturbed = None
        robustness_drop = None
        if args.perturb:
            # Make perturbation deterministic per-sample.
            state = random.getstate()
            random.seed(args.seed + i)
            perturbed_doc = perturb_docstring(sample["docstring"])
            random.setstate(state)

            prompt_p = _format_prompt(prompt_template, sample["function_name"], perturbed_doc)
            completion_p = model.generate(prompt_p, max_tokens=args.max_new_tokens)
            gen_code_p = extract_code(completion_p)
            minhash_perturbed = calculate_minhash_similarity(gen_code_p, ref_code)
            robustness_drop = minhash - minhash_perturbed

        results.append(
            {
                "function_name": sample["function_name"],
                "original_minhash": float(minhash),
                "original_exact": bool(exact),
                "perturbed_minhash": None if minhash_perturbed is None else float(minhash_perturbed),
                "robustness_drop": None if robustness_drop is None else float(robustness_drop),
            }
        )

        generated_codes.append(gen_code)
        reference_codes.append(ref_code)

    avg_sim = float(np.mean([r["original_minhash"] for r in results])) if results else 0.0
    avg_robust_sim = (
        float(np.mean([r["perturbed_minhash"] for r in results if r["perturbed_minhash"] is not None]))
        if results and args.perturb
        else None
    )
    exact_matches = int(sum(1 for r in results if r["original_exact"]))
    robustness_drop_avg = (
        float(np.mean([r["robustness_drop"] for r in results if r["robustness_drop"] is not None]))
        if results and args.perturb
        else None
    )

    bg_ref = _background_similarity(reference_codes, seed=args.seed)
    bg_gen = _background_similarity(generated_codes, seed=args.seed)

    summary = {
        "n": len(results),
        "avg_original_minhash": avg_sim,
        "avg_perturbed_minhash": avg_robust_sim,
        "exact_matches": exact_matches,
        "exact_match_rate": (exact_matches / len(results)) if results else 0.0,
        "avg_robustness_drop": robustness_drop_avg,
        "background_ref_minhash": bg_ref,
        "background_gen_minhash": bg_gen,
        "high_similarity_rate@0.7": (
            float(np.mean([1.0 if r["original_minhash"] >= 0.7 else 0.0 for r in results])) if results else 0.0
        ),
    }

    report = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "perturb": bool(args.perturb),
            "prompt": args.prompt,
        },
        "summary": summary,
        "results": results,
        "interpretation": {
            "how_to_read": (
                "- original_minhash misura quanto il codice generato assomiglia alla reference (0=diverso, 1=uguale).\n"
                "- original_exact=True indica copia verbatim (se l'estrazione del codice è corretta).\n"
                "- perturbed_minhash ripete la misura dopo una perturbazione della docstring.\n"
                "- robustness_drop = original_minhash - perturbed_minhash: se è grande, il comportamento è più 'trigger-based' (proxy di memorization).\n"
                "- background_* sono similarità attese tra esempi non-correlati (baseline): se original_minhash è vicino al background, non c'è segnale forte di recall."  # noqa: E501
            ),
            "caveat": (
                "Questo esperimento non può provare in modo definitivo che il modello abbia visto CodeSearchNet nel training; "
                "mostra invece segnali compatibili con memorization/leakage (alta similarità + fragilità a perturbazioni) "
                "vs generalizzazione robusta (similarità stabile anche con perturbazioni)."
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe LLM memorization on SE datasets (default: CodeSearchNet-Python).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="prompts/code_gen_prompt.txt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--perturb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    report = run(args)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    s = report["summary"]
    print(f"n={s['n']} | avg_minhash={s['avg_original_minhash']:.4f} | exact={s['exact_matches']}/{s['n']}")
    if s.get("avg_perturbed_minhash") is not None:
        print(
            f"avg_perturbed={s['avg_perturbed_minhash']:.4f} | avg_drop={s['avg_robustness_drop']:.4f} | high_sim@0.7={s['high_similarity_rate@0.7']:.2%}"
        )
    print(f"background_ref={s['background_ref_minhash']:.4f} | background_gen={s['background_gen_minhash']:.4f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
