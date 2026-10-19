"""Integration checks run on the downloaded corpus/bundle, never synthetic training data."""
import json
from pathlib import Path
import numpy as np
import pytest
from clinical_ade.data import assert_disjoint
from clinical_ade.service import Engine,sentences

ROOT=Path(__file__).resolve().parents[1]


def test_real_pinned_corpus_and_span_alignment():
    file=ROOT/'runtime/prepared.json'
    if not file.exists():pytest.skip('Run ade-prepare for real-data integration')
    rows=json.loads(file.read_text());assert_disjoint(rows)
    assert len(rows)==20895
    assert len({p for r in rows for p in r['pmids']})==2972
    assert len([r for r in rows if r['split']=='test'])==4172
    for row in rows:
        assert all(p.isdigit() for p in row['pmids'])
        assert row['label'] in [0,1]
        for e in row['entities']:assert 0<=e['start']<e['end']<=len(row['text'])


def test_trained_inference_matches_saved_test_probabilities():
    if not (ROOT/'runtime/artifacts/manifest.json').exists():pytest.skip('Run training for bundle integration')
    engine=Engine(ROOT)
    rows=json.loads((ROOT/'runtime/test_predictions.json').read_text())
    candidates=[r for r in rows if len(list(sentences(r['text'])))==1][:5]
    assert len(candidates)==5
    for row in candidates:
        result=engine.analyze(row['text'])
        assert abs(result['sentences'][0]['ade_probability']-row['predictions'][engine.name])<1e-4
        for s in result['sentences']:
            assert row['text'][s['start']:s['end']]==s['text']
            for e in s['entities']:assert row['text'][e['start']:e['end']]==e['text']


def test_promoted_neural_entities_match_saved_predictions():
    manifest=ROOT/'runtime/artifacts/manifest.json'
    if not manifest.exists() or json.loads(manifest.read_text()).get('selected_extractor')!='biomedbert_token':
        pytest.skip('Run token-extractor training and promotion for neural bundle integration')
    engine=Engine(ROOT)
    rows=json.loads((ROOT/'runtime/test_predictions.json').read_text())
    chosen=[r for r in rows if r['predictions'][engine.name]>=engine.threshold
            and len(list(sentences(r['text'])))==1][:3]
    assert len(chosen)==3
    for row in chosen:
        out=engine.analyze(row['text'])
        assert out['entity_extractor']=='biomedbert_token'
        assert out['sentences'][0]['entities']==row['entities']
