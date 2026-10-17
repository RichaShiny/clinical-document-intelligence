"""Corpus exploration and local model run provenance."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import platform
import statistics
from pathlib import Path
from urllib.request import urlopen
from .dataset import corpus


def describe(rows):
    lengths = [len(r['text']) for r in rows]
    return dict(records=len(rows), characters=dict(min=min(lengths),median=statistics.median(lengths),max=max(lengths)),
        splits=dict(Counter(r['split'] for r in rows)),
        assertions=dict(Counter(e['status'] for r in rows for e in r['gold'])),
        entity_types=dict(Counter(e['kind'] for r in rows for e in r['gold'])),
        unique_texts=len({r['text'] for r in rows}),
        note='Repeated sentence templates across synthetic groups are intentional. No real population inference is valid.')


def capture_environment(root, model):
    from .llm import PROMPT, SCHEMA
    with urlopen('http://127.0.0.1:11434/api/tags',timeout=10) as response:
        tags=json.load(response)
    with urlopen('http://127.0.0.1:11434/api/version',timeout=10) as response:
        version=json.load(response)
    entry=next((m for m in tags['models'] if m['name']==model),None)
    if entry is None:
        raise ValueError('Use an installed exact Ollama model tag from /api/tags')
    provenance=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),model=entry,
        ollama=version, python=platform.python_version(), platform=platform.platform(),
        machine=platform.machine(),temperature=0,seed=42,
        prompt_schema_sha256=hashlib.sha256((PROMPT+json.dumps(SCHEMA,sort_keys=True)).encode()).hexdigest(),
        note='Prompt plus response-schema hash; vocabulary and source are versioned in Git. No hardware energy cost measured.')
    (root/'reports').mkdir(exist_ok=True)
    (root/'reports'/'llm_environment.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--ollama-model')
    args=parser.parse_args()
    (args.root/'reports').mkdir(exist_ok=True)
    description=describe(corpus())
    (args.root/'reports'/'eda.json').write_text(json.dumps(description,indent=2)+'\n')
    if args.ollama_model: capture_environment(args.root,args.ollama_model)
    print(json.dumps(description,indent=2))

if __name__=='__main__':main()
