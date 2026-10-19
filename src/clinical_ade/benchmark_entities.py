"""Compare extractor latency via the same in-process ASGI API protocol.

Uses isolated temporary manifests and review stores; never changes the serving bundle.
Network transport and startup are excluded. All inputs are real local held-out sentences.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
from unittest.mock import patch
from fastapi.testclient import TestClient
from .api import create_app
from .train import write_json


def benchmark(root):
    root=Path(root)
    source=root/'runtime/artifacts'
    manifest=json.loads((source/'manifest.json').read_text())
    rows=json.loads((root/'runtime/test_predictions.json').read_text())[:23]
    results={}
    for choice in ['crf','biomedbert_token']:
        with tempfile.TemporaryDirectory() as temp:
            isolated=Path(temp)
            directory=isolated/'runtime/artifacts'
            directory.mkdir(parents=True)
            for file in source.iterdir():
                if file.is_file() and file.name!='manifest.json':
                    (directory/file.name).symlink_to(file.resolve())
            (isolated/'runtime/biomedbert').symlink_to((root/'runtime/biomedbert').resolve())
            configuration=copy.deepcopy(manifest)
            configuration['selected_extractor']=choice
            write_json(directory/'manifest.json',configuration)
            durations=[]
            with patch.dict(os.environ,{'ADE_REVIEW_DB':str(isolated/'reviews.sqlite')}), \
                    TestClient(create_app(isolated,token='ephemeral-benchmark')) as client:
                assert client.get('/health/ready').status_code==200
                for i,row in enumerate(rows):
                    start=time.perf_counter()
                    response=client.post('/v1/analyze',headers={'Authorization':'Bearer ephemeral-benchmark'},
                                         json={'text':row['text']})
                    elapsed=(time.perf_counter()-start)*1000
                    response.raise_for_status()
                    result=response.json()
                    assert result['entity_extractor']==choice
                    for sentence in result['sentences']:
                        assert row['text'][sentence['start']:sentence['end']]==sentence['text']
                        for entity in sentence['entities']:
                            assert row['text'][entity['start']:entity['end']]==entity['text']
                    if i>=3:durations.append(elapsed)
                results[choice]=dict(n=len(durations),median_ms=statistics.median(durations),
                    p95_ms=sorted(durations)[int(.95*(len(durations)-1))],model_version=result['model_version'])
    report=dict(protocol='Same 20 real test sentences after 3 warmups; in-process FastAPI ASGI requests. No network transport or startup. CPU / four PyTorch threads.',
                models=results,limitations='Small local latency sample, not a load test. Most sentences may bypass extraction at the fixed triage threshold.')
    write_json(root/'reports/entity_inference_benchmark.json',report)
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    benchmark(parser.parse_args().root)

if __name__=='__main__':main()
