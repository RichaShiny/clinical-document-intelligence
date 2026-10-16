"""Exact span + concept + assertion evaluation and paired note bootstrap."""
import random
from collections import Counter


def key(e):
    return (e["kind"], e["value"], e["status"], e["start"], e["end"])


def counts(gold, predicted):
    g, p = set(map(key, gold)), set(map(key, predicted))
    return len(g & p), len(p - g), len(g - p)


def metrics(totals):
    tp, fp, fn = totals
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def aggregate(items):
    return metrics(tuple(sum(c[i] for c in items) for i in range(3)))


def bootstrap(items, samples=500):
    rng = random.Random(42)
    estimates = sorted(aggregate(rng.choices(items, k=len(items)))["f1"] for _ in range(samples))
    return [estimates[int(samples * .025)], estimates[int(samples * .975)]]


def evaluate(rows, predictions):
    report = {}
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        c = [counts(r["gold"], predictions[r["id"]]) for r in subset]
        report[split] = dict(n=len(subset), **aggregate(c), f1_ci95=bootstrap(c),
            by_kind={k: aggregate([counts([e for e in r["gold"] if e["kind"] == k],
                [e for e in predictions[r["id"]] if e["kind"] == k]) for r in subset])
                for k in ["condition", "medication", "procedure", "follow_up"]},
            by_age={a: dict(n=sum(r["age_group"] == a for r in subset),
                **aggregate([counts(r["gold"], predictions[r["id"]]) for r in subset if r["age_group"] == a]))
                for a in sorted({r["age_group"] for r in subset})},
            assertion_confusion=dict(Counter(
                e["status"] + " → " + p["status"]
                for r in subset for e in r["gold"] for p in predictions[r["id"]]
                if (e["kind"],e["start"],e["end"]) == (p["kind"],p["start"],p["end"]))))
    return report


def paired_delta(rows, first, second, split="test"):
    subset = [r for r in rows if r["split"] == split]
    a = [counts(r["gold"], first[r["id"]]) for r in subset]
    b = [counts(r["gold"], second[r["id"]]) for r in subset]
    rng = random.Random(42)
    deltas = []
    for _ in range(500):
        idx = rng.choices(range(len(subset)), k=len(subset))
        deltas.append(aggregate([b[i] for i in idx])["f1"] - aggregate([a[i] for i in idx])["f1"])
    deltas.sort()
    return dict(split=split, delta_f1=aggregate(b)["f1"]-aggregate(a)["f1"],
                ci95=[deltas[12], deltas[487]], bootstrap_samples=500)
