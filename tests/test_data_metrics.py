import numpy as np
import pytest
from clinical_ade.data import normalize,UnionFind,assert_disjoint
from clinical_ade.metrics import choose_threshold,calibrator,calibrated,evaluate,cluster_bootstrap,reliability
from clinical_ade import entities,neural
from clinical_ade.service import sentences


def test_duplicate_documents_connect_transitively():
    uf=UnionFind();uf.union('1','2');uf.union('2','3')
    assert uf.find('1')==uf.find('3')


def test_document_leakage_fails_closed():
    rows=[dict(pmids=['1'],text='sentence A',split='train'),dict(pmids=['1'],text='sentence B',split='test')]
    with pytest.raises(ValueError,match='Document'):assert_disjoint(rows)


def test_normalized_duplicate_leakage_fails_closed():
    rows=[dict(pmids=['1'],text='  Duplicate TEXT ',split='train'),dict(pmids=['2'],text='duplicate text',split='test')]
    with pytest.raises(ValueError,match='Text'):assert_disjoint(rows)


def test_threshold_targets_recall_on_policy_data():
    y=np.array([1,1,1,0,0]);p=np.array([.9,.8,.1,.7,.05])
    threshold=choose_threshold(y,p,target=1.)
    assert threshold==.1
    assert ((p>=threshold)&(y==1)).sum()==3


def test_calibration_and_metrics_are_finite():
    y=np.array([0,1,0,1,0,1]);p=np.array([0.,1.,.2,.6,.3,.9])
    model=calibrator(y,p);out=calibrated(model,p)
    assert np.isfinite(out).all()
    result=evaluate(y,out,.5)
    assert sum(map(sum,result['confusion_matrix']))==len(y)
    assert sum(b['n'] for b in result['reliability'])==len(y)


def test_reliability_includes_probability_one_and_bin_edges():
    ece,bins=reliability(np.array([0,1,1]),np.array([0.,.3,1.]))
    assert sum(b['n'] for b in bins)==3
    assert 0<=ece<=1


def test_cluster_bootstrap_is_reproducible():
    y=[0,1,0,1];p=[.1,.8,.2,.9];g=['a','a','b','b']
    assert cluster_bootstrap(y,p,g,.5,samples=30)==cluster_bootstrap(y,p,g,.5,samples=30)


def test_crf_bio_alignment_and_exact_offsets():
    row=dict(text='aspirin caused rash.',entities=[dict(kind='DRUG',start=0,end=7),dict(kind='EFFECT',start=15,end=19)])
    features,tags,overlap=entities.sample(row)
    spans=entities.decode(row['text'],tags)
    assert tags==['B-DRUG','O','B-EFFECT','O']
    assert spans[1]['text']=='rash'
    assert row['text'][spans[0]['start']:spans[0]['end']]=='aspirin'


def test_sentence_offsets_are_verbatim():
    text='  First sentence.\n Second sentence!'
    parts=list(sentences(text))
    assert len(parts)==2
    for start,end,sentence in parts:assert text[start:end]==sentence


def test_neural_empty_and_unseen_text_is_supported():
    vocab=neural.vocabulary(['known known token token'])
    rows=neural.encode(['','unseen'],vocab)
    model=neural.DeepAverage(len(vocab)+1)
    assert neural.predict(model,rows).shape==(2,)
