"""Generate data, execute systems, evaluate, and export the dashboard dataset."""
import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path
from .dataset import write_data
from .evaluate import evaluate, paired_delta, key
from .extract import extract, summarize


def run(root, model=None):
    rows, manifest = write_data(root / "data")
    from .analysis import describe, capture_environment
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "eda.json").write_text(json.dumps(describe(rows), indent=2) + "\n")
    if model:
        capture_environment(root, model)
    systems = {"dictionary": lambda text: extract(text, False),
               "contextual": lambda text: extract(text, True)}
    results, all_predictions, details = {}, {}, {r["id"]: {} for r in rows}
    if model:
        from .llm import extract_llm
        systems["ollama:" + model] = lambda text: extract_llm(text, model)
    for name, predict in systems.items():
        system_rows = rows
        if name.startswith("ollama:"):
            # Fixed sample, selected without looking at model outputs.
            test_rows = [r for r in rows if r["split"] == "test"]
            system_rows = random.Random(42).sample(test_rows, 15) + [r for r in rows if r["split"] == "challenge"]
        predictions, times, rejected, tokens = {}, [], 0, 0
        for row in system_rows:
            start = time.perf_counter()
            output = predict(row["text"])
            elapsed = (time.perf_counter() - start) * 1000
            extra = {}
            if isinstance(output, tuple):
                output, extra = output
                rejected += len(extra["rejected"])
                tokens += (extra.get("prompt_tokens") or 0) + (extra.get("output_tokens") or 0)
            predictions[row["id"]] = output
            times.append(elapsed)
            if name.startswith("ollama:"):
                print(f"LLM {len(times)}/{len(system_rows)}: {row['id']} ({elapsed/1000:.1f}s)", flush=True)
            g, p = {key(e) for e in row["gold"]}, {key(e) for e in output}
            details[row["id"]][name] = dict(entities=output, summary=summarize(row["text"], output),
                false_positives=[list(e) for e in sorted(p-g)],
                false_negatives=[list(e) for e in sorted(g-p)], **extra)
        results[name] = dict(splits=evaluate(system_rows,predictions), evaluated_note_ids=[r["id"] for r in system_rows],
            latency_ms=dict(median=statistics.median(times),p95=sorted(times)[int(len(times)*.95)]),
            rejected_entities=rejected, tokens=tokens,
            summary_method="extractive source sentences; no abstractive generation",
            api_cost_usd=0, hardware_cost_usd=None)
        all_predictions[name] = predictions
    llm_comparison = None
    if model:
        name = "ollama:" + model
        subset = [r for r in rows if r["id"] in all_predictions[name]]
        llm_comparison = dict(baseline_on_same_subset=evaluate(subset,all_predictions["contextual"]),
            paired_test_delta=paired_delta(subset,all_predictions["contextual"],all_predictions[name]))
    report = dict(llm_comparison=llm_comparison, dataset=manifest, python=platform.python_version(),
        llm_status="executed" if model else "not run — optional Ollama model required",
        systems=results, paired_comparison=paired_delta(rows,all_predictions["dictionary"],all_predictions["contextual"]),
        metric="Exact kind, canonical value, assertion, start and end; micro F1",
        limitations=["Template-generated benchmark is not clinical validation.",
                      "Age groups are synthetic metadata; slice parity does not establish fairness.",
                      "Citation matching does not establish semantic correctness."])
    (root/"reports").mkdir(exist_ok=True)
    (root/"reports"/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    payload = dict(report=report, notes=[dict(r, predictions=details[r["id"]]) for r in rows])
    encoded = json.dumps(payload,ensure_ascii=False)
    (root/"dashboard"/"results.json").write_text(encoded+"\n")
    # Buildless self-contained dashboard, also works from file:// with no fetch server.
    (root/"dashboard"/"results.js").write_text("window.CLINICAL_DATA = " + encoded.replace("<","\\u003c") + ";\n")
    digest = hashlib.sha256((root/"data"/"synthetic_notes.jsonl").read_bytes()).hexdigest()
    assert digest == manifest["sha256"]
    print(json.dumps({"records":len(rows),"llm_status":report["llm_status"],
                      "systems":{n:{s:round(v["f1"],3) for s,v in r["splits"].items()} for n,r in results.items()}},indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ollama-model",help="Explicit local model tag; runs all notes and reports actual results")
    args = parser.parse_args()
    run(args.root,args.ollama_model)

if __name__ == "__main__":
    main()
