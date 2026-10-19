"""Offset alignment and exact-span evaluation without downloading model weights."""
from clinical_ade.token_entities import align_labels, decode_offsets, span_metrics


def test_subwords_receive_bio_labels_and_specials_are_ignored():
    offsets=[(0,0),(0,3),(3,7),(8,13),(0,0)]
    labels,ignored=align_labels(offsets,[dict(kind='DRUG',start=0,end=7)])
    assert labels==[-100,1,2,0,-100]
    assert ignored==0


def test_truncated_entity_is_not_supervised_as_complete_or_negative():
    labels,ignored=align_labels([(0,0),(0,4),(4,7),(0,0)],
                                [dict(kind='EFFECT',start=0,end=12)])
    assert labels==[-100,-100,-100,-100]
    assert ignored==1


def test_partial_token_boundary_is_ignored():
    labels,ignored=align_labels([(0,0),(0,7),(0,0)],[dict(kind='DRUG',start=1,end=7)])
    assert labels==[-100,-100,-100]
    assert ignored==1


def test_overlapping_span_does_not_overwrite_first_label():
    labels,ignored=align_labels([(0,0),(0,3),(3,7),(0,0)],
        [dict(kind='DRUG',start=0,end=7),dict(kind='EFFECT',start=3,end=7)])
    assert labels==[-100,1,2,-100]
    assert ignored==1


def test_decode_joins_subwords_and_preserves_exact_source_offsets():
    text='aspirin caused skin rash'
    offsets=[(0,0),(0,3),(3,7),(8,14),(15,19),(20,24),(0,0)]
    spans=decode_offsets(text,offsets,['O','B-DRUG','I-DRUG','O','B-EFFECT','I-EFFECT','O'])
    assert spans==[dict(kind='DRUG',start=0,end=7,text='aspirin'),
                   dict(kind='EFFECT',start=15,end=24,text='skin rash')]


def test_orphan_inside_tag_starts_new_entity():
    assert decode_offsets('rash',[(0,4)],['I-EFFECT'])[0]['text']=='rash'


def test_exact_span_metrics_penalize_type_errors_and_apply_gating():
    rows=[dict(id='a',entities=[dict(kind='DRUG',start=0,end=7)])]
    predicted=[[dict(kind='EFFECT',start=0,end=7)]]
    wrong=span_metrics(rows,predicted)
    assert (wrong['tp'],wrong['fp'],wrong['fn'])==(0,1,1)
    correct=[[dict(kind='DRUG',start=0,end=7)]]
    assert span_metrics(rows,correct)['micro_f1']==1
    assert span_metrics(rows,correct,gates={'a':False})['fn']==1


def test_serving_rejects_unregistered_neural_checkpoint(tmp_path):
    import json
    import pytest
    from clinical_ade.service import Engine
    directory=tmp_path/'runtime/artifacts'
    directory.mkdir(parents=True)
    (directory/'manifest.json').write_text(json.dumps(dict(
        selected_model='tfidf_logistic',selected_extractor='biomedbert_token',files={})))
    with pytest.raises(ValueError,match='not registered'):Engine(tmp_path)


def test_serving_rejects_unknown_extractor(tmp_path):
    import json
    import pytest
    from clinical_ade.service import Engine
    directory=tmp_path/'runtime/artifacts'
    directory.mkdir(parents=True)
    (directory/'manifest.json').write_text(json.dumps(dict(
        selected_model='tfidf_logistic',selected_extractor='unknown',files={})))
    with pytest.raises(ValueError,match='Unsupported entity'):Engine(tmp_path)


def test_evaluation_rejects_missing_predictions():
    import pytest
    with pytest.raises(ValueError,match='Prediction count'):
        span_metrics([dict(id='a',entities=[])],[])
    with pytest.raises(ValueError,match='length mismatch'):
        decode_offsets('rash',[(0,4)],[])


def test_paired_document_bootstrap_is_reproducible():
    from clinical_ade.token_entities import paired_span_delta
    rows=[dict(id='a',group='doc1',entities=[dict(kind='DRUG',start=0,end=7)])]
    baseline=[[]]
    candidate=[[dict(kind='DRUG',start=0,end=7)]]
    first=paired_span_delta(rows,baseline,candidate,samples=20)
    assert first['delta_f1']==1
    assert first['ci95']==[1,1]
    assert first==paired_span_delta(rows,baseline,candidate,samples=20)


def test_serving_neural_bundle_requires_encoder_digests(tmp_path):
    import hashlib
    import json
    import pytest
    from clinical_ade.service import Engine
    directory=tmp_path/'runtime/artifacts'
    directory.mkdir(parents=True)
    delta=directory/'entities_biomedbert_delta.pt'
    delta.write_bytes(b'not-loaded')
    manifest=dict(selected_model='tfidf_logistic',selected_extractor='biomedbert_token',
                  files={delta.name:hashlib.sha256(delta.read_bytes()).hexdigest()})
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='encoder files are not registered'):Engine(tmp_path)
