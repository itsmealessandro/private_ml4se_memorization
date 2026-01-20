"""Run a small memorization probe for code-generation LLMs.

This script:
- loads a SE/code dataset (default: CodeSearchNet Python) and samples N examples,
- formats a prompt from a template using each example's function name + docstring,
- asks a local model wrapper to generate code,
- extracts code from the completion and compares it to the reference using MinHash
    similarity and an exact-match check,
- optionally perturbs the docstring and re-runs generation to estimate robustness
    (a drop in similarity after perturbation is used as a memorization proxy),
- writes a JSON report with per-sample results plus summary statistics.
"""

import argparse
import json
import random
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from evaluation.memorization_metrics import calculate_minhash_similarity, check_exact_match
from evaluation.output_parsing import extract_code
from evaluation.robustness import perturb_docstring
from models.model_wrappers import LocalLLM


DEFAULT_DATASET = "Nan-Do/code-search-net-python"
DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"


# Fixed decision thresholds used for the human-readable report.
# You can tweak these manually if you want stricter/looser verdicts.
T_EXACT_STRONG = 0.05
T_MARGIN_STRONG = 0.20
T_HIGH_SIM_STRONG = 0.20
T_DROP_STRONG = 0.10

T_EXACT_LOW = 0.01
T_MARGIN_LOW = 0.10
T_DROP_LOW = 0.05


def _make_run_dir(base_dir: str | Path) -> Path:
    # Each analysis run gets its own folder: analysis_<timestamp>
    # Uses Europe/Rome timezone with format: YYYYMMDD_HHMM (no seconds)
    rome_tz = timezone(timedelta(hours=1))  # CET (use +2 for CEST if needed)
    now_rome = datetime.now(rome_tz)
    run_id = now_rome.strftime("%Y%m%d_%H%M")
    run_dir = Path(base_dir) / f"analysis_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _start_heartbeat(message_fn, interval_s: float = 5.0) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _loop() -> None:
        # Print periodically to show liveness even if generation is slow.
        while not stop_event.wait(interval_s):
            tqdm.write(message_fn(), file=sys.stderr)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return stop_event, t


def _stop_heartbeat(stop_event: threading.Event, thread: threading.Thread) -> None:
    stop_event.set()
    thread.join(timeout=1.0)


def _verdict_from_summary(summary: dict, thresholds: dict) -> tuple[str, str]:
    n = int(summary.get("n", 0))
    if n <= 0:
        return "INCONCLUSIVE", "No samples were evaluated."

    exact_rate = float(summary.get("exact_match_rate", 0.0))
    avg_minhash = float(summary.get("avg_original_minhash", 0.0))
    high_sim = float(summary.get("high_similarity_rate@0.7", 0.0))
    bg_ref = float(summary.get("background_ref_minhash", 0.0))
    margin = avg_minhash - bg_ref

    avg_drop = summary.get("avg_robustness_drop", None)
    avg_drop_f = None if avg_drop is None else float(avg_drop)

    strong = (
        (exact_rate >= thresholds["T_EXACT_STRONG"])
        or (
            (margin >= thresholds["T_MARGIN_STRONG"])
            and (high_sim >= thresholds["T_HIGH_SIM_STRONG"])
            and (
                (avg_drop_f is None)
                or (avg_drop_f >= thresholds["T_DROP_STRONG"])
            )
        )
    )
    if strong:
        if exact_rate >= thresholds["T_EXACT_STRONG"]:
            return (
                "STRONG_EVIDENCE_MEMORIZATION",
                f"Exact-match rate is high ({exact_rate:.1%}), suggesting verbatim reproduction.",
            )
        drop_part = "" if avg_drop_f is None else f" and robustness drop is sizable ({avg_drop_f:.3f})"
        return (
            "STRONG_EVIDENCE_MEMORIZATION",
            f"Similarity is well above background (margin={margin:.3f}), high-sim rate is elevated ({high_sim:.1%}){drop_part}.",
        )

    little = (
        (margin < thresholds["T_MARGIN_LOW"])
        and (exact_rate < thresholds["T_EXACT_LOW"])
        and (
            (avg_drop_f is None)
            or (avg_drop_f < thresholds["T_DROP_LOW"])
        )
    )
    if little:
        return (
            "LITTLE_EVIDENCE_MEMORIZATION",
            f"Similarity is close to background (margin={margin:.3f}) and exact-match rate is low ({exact_rate:.1%}).",
        )

    return "INCONCLUSIVE", "Signals are mixed; run more samples or inspect per-sample outputs."


def _render_human_report(report: dict, thresholds: dict, results_path: Path) -> str:
    meta = report.get("meta", {})
    summary = report.get("summary", {})

    verdict, rationale = _verdict_from_summary(summary, thresholds)

    lines: list[str] = []
    lines.append("# LLM Memorization Probe Report")
    lines.append("")
    lines.append("## Run Info")
    lines.append(f"- Timestamp (UTC): {meta.get('timestamp', '')}")
    lines.append(f"- Model: {meta.get('model', '')}")
    lines.append(f"- Dataset: {meta.get('dataset', '')} / split={meta.get('split', '')}")
    lines.append(f"- Samples (n): {summary.get('n', 0)}")
    lines.append(f"- Prompt template: {meta.get('prompt', '')}")
    lines.append(f"- Docstring perturbation: {bool(meta.get('perturb', False))}")
    lines.append(f"- Seed: {meta.get('seed', '')}")
    lines.append(f"- Max new tokens: {meta.get('max_new_tokens', '')}")
    lines.append("")

    lines.append("## Executive Summary (Human-readable)")
    lines.append(f"- Verdict: {verdict}")
    lines.append(f"- One-line rationale: {rationale}")
    lines.append("")

    avg_orig = float(summary.get("avg_original_minhash", 0.0))
    exact_matches = int(summary.get("exact_matches", 0))
    exact_rate = float(summary.get("exact_match_rate", 0.0))
    high_sim = float(summary.get("high_similarity_rate@0.7", 0.0))
    avg_pert = summary.get("avg_perturbed_minhash", None)
    avg_drop = summary.get("avg_robustness_drop", None)
    bg_ref = float(summary.get("background_ref_minhash", 0.0))
    bg_gen = float(summary.get("background_gen_minhash", 0.0))
    margin = avg_orig - bg_ref

    lines.append("## Key Metrics")
    lines.append(f"- Avg MinHash similarity (original): {avg_orig:.4f}")
    lines.append(f"- Exact match rate (original): {exact_rate:.2%}  (exact_matches={exact_matches}/{summary.get('n', 0)})")
    lines.append(f"- High similarity rate @0.7: {high_sim:.2%}")
    lines.append(f"- Background similarity (reference baseline): {bg_ref:.4f}")
    lines.append(f"- Background similarity (generated baseline): {bg_gen:.4f}")
    lines.append(f"- Similarity margin vs background (avg_original - background_ref): {margin:.4f}")
    if avg_pert is not None:
        lines.append(f"- Avg MinHash similarity (perturbed): {float(avg_pert):.4f}")
    if avg_drop is not None:
        lines.append(f"- Avg robustness drop (original - perturbed): {float(avg_drop):.4f}")
    lines.append("")

    lines.append("## Decision Criteria (What ‘verdict’ means)")
    lines.append("We label the run as:")
    lines.append("- STRONG_EVIDENCE_MEMORIZATION if:")
    lines.append(f"  - exact_match_rate >= {thresholds['T_EXACT_STRONG']:.2f}")
    lines.append("  OR")
    lines.append(
        f"  - (avg_original_minhash - background_ref_minhash) >= {thresholds['T_MARGIN_STRONG']:.2f} AND high_similarity_rate@0.7 >= {thresholds['T_HIGH_SIM_STRONG']:.2f}"
        + (" AND avg_robustness_drop >= %.2f" % thresholds["T_DROP_STRONG"] if avg_drop is not None else "")
    )
    lines.append("- LITTLE_EVIDENCE_MEMORIZATION if all of:")
    lines.append(f"  - (avg_original_minhash - background_ref_minhash) < {thresholds['T_MARGIN_LOW']:.2f}")
    lines.append(f"  - exact_match_rate < {thresholds['T_EXACT_LOW']:.2f}")
    if avg_drop is not None:
        lines.append(f"  - avg_robustness_drop < {thresholds['T_DROP_LOW']:.2f}")
    lines.append("- otherwise INCONCLUSIVE.")
    lines.append("")

    lines.append("## Interpretation Notes")
    lines.append("- Exact match suggests verbatim reproduction (subject to code-extraction correctness).")
    lines.append("- MinHash close to 1 indicates strong textual overlap; compare against background baselines.")
    if avg_drop is not None:
        lines.append("- A large robustness drop after docstring perturbation suggests trigger sensitivity (memorization-like) vs robust generalization.")
    lines.append("")

    lines.append("## Files Produced")
    lines.append(f"- results.json: {results_path.name} (machine-readable)")
    lines.append("- REPORT.md: this document")
    lines.append("")

    return "\n".join(lines)


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

    if getattr(args, "streaming", False):
        dataset = load_dataset(args.dataset, split=args.split, streaming=True)
        buffer_size = int(getattr(args, "streaming_buffer_size", 10_000))
        if buffer_size > 1:
            dataset = dataset.shuffle(seed=args.seed, buffer_size=buffer_size)
        dataset = list(dataset.take(args.n))
    else:
        dataset = load_dataset(args.dataset, split=args.split)
        dataset = dataset.shuffle(seed=args.seed)
        # Guard against args.n > len(dataset)
        n = min(args.n, len(dataset))
        dataset = dataset.select(range(n))

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

    total = len(samples)
    pbar = tqdm(samples, desc="Probing", total=total)
    for i, sample in enumerate(pbar):
        # `tqdm` shows overall percent; heartbeat covers long per-sample generation.
        pbar.set_postfix_str(f"{i + 1}/{total}")
        prompt = _format_prompt(prompt_template, sample["function_name"], sample["docstring"])
        start_t = time.time()

        def _msg_orig() -> str:
            pct = (100.0 * (i) / total) if total else 0.0
            return f"[alive] {i + 1}/{total} ({pct:5.1f}%) generating (original) for {time.time() - start_t:.0f}s"

        hb_stop, hb_thread = _start_heartbeat(_msg_orig)
        try:
            completion = model.generate(prompt, max_tokens=args.max_new_tokens)
        finally:
            _stop_heartbeat(hb_stop, hb_thread)
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
            start_t_p = time.time()

            def _msg_pert() -> str:
                pct = (100.0 * (i) / total) if total else 0.0
                return f"[alive] {i + 1}/{total} ({pct:5.1f}%) generating (perturbed) for {time.time() - start_t_p:.0f}s"

            hb_stop_p, hb_thread_p = _start_heartbeat(_msg_pert)
            try:
                completion_p = model.generate(prompt_p, max_tokens=args.max_new_tokens)
            finally:
                _stop_heartbeat(hb_stop_p, hb_thread_p)
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
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load dataset in streaming mode (recommended on Colab to avoid downloading full splits).",
    )
    parser.add_argument(
        "--streaming-buffer-size",
        type=int,
        default=10_000,
        help="Shuffle buffer size used only with --streaming (higher = better shuffle, more RAM).",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Output JSON filename (it will be placed under analysis_results/<run_id>/).",
    )
    args = parser.parse_args()

    run_dir = _make_run_dir("reports")
    out_name = Path(args.output).name
    results_path = run_dir / out_name

    report = run(args)
    # Store run location in the JSON meta for convenience.
    report.setdefault("meta", {})["run_dir"] = str(run_dir)
    report["meta"]["results_json"] = str(results_path)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    thresholds = {
        "T_EXACT_STRONG": T_EXACT_STRONG,
        "T_MARGIN_STRONG": T_MARGIN_STRONG,
        "T_HIGH_SIM_STRONG": T_HIGH_SIM_STRONG,
        "T_DROP_STRONG": T_DROP_STRONG,
        "T_EXACT_LOW": T_EXACT_LOW,
        "T_MARGIN_LOW": T_MARGIN_LOW,
        "T_DROP_LOW": T_DROP_LOW,
    }
    human_report = _render_human_report(report, thresholds=thresholds, results_path=results_path)
    report_path = run_dir / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(human_report)

    s = report["summary"]
    print(f"n={s['n']} | avg_minhash={s['avg_original_minhash']:.4f} | exact={s['exact_matches']}/{s['n']}")
    if s.get("avg_perturbed_minhash") is not None:
        print(
            f"avg_perturbed={s['avg_perturbed_minhash']:.4f} | avg_drop={s['avg_robustness_drop']:.4f} | high_sim@0.7={s['high_similarity_rate@0.7']:.2%}"
        )
    print(f"background_ref={s['background_ref_minhash']:.4f} | background_gen={s['background_gen_minhash']:.4f}")
    print(f"Wrote {results_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
