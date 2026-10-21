import json
import pytest
from clinical_ade.error_analysis import analyze


def row(text, label, score, gold=None, predicted=None):
    return dict(text=text, label=label, predictions={'model': score},
                gold_entities=gold or [], entities=predicted or [])


def test_boundaries_gating_and_aggregate_only_output():
    entity = dict(kind='DRUG', start=0, end=1)
    rows = [row('a'*160, 1, .5, [entity], [entity]),
            row('b'*161, 1, .49, [entity], [entity]),
            row('c'*320, 0, .8), row('d'*321, 0, .1)]
    report = analyze(rows, 'model', .5)
    assert [report[k]['n'] for k in list(report)[1:]] == [1, 2, 1]
    total = report['all']
    assert (total['tp'], total['fn'], total['fp'], total['tn']) == (1, 1, 1, 1)
    assert total['gated_exact_spans']['DRUG'] == dict(tp=1, fp=0, fn=1, f1=2/3)
    assert 'a'*160 not in json.dumps(report)


def test_empty_slice_has_undefined_rates():
    report = analyze([row('text', 0, .1)], 'model', .5)
    assert report['all']['recall'] is None
    assert report['chars_321_plus']['review_fraction'] is None


@pytest.mark.parametrize('rows', [[], [row('text', 1, float('nan'))], [row('text', 2, .4)]])
def test_invalid_predictions_fail(rows):
    with pytest.raises(ValueError):
        analyze(rows, 'model', .5)
