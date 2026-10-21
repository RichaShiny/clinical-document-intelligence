from clinical_ade.service import token_coverage


class Tokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs == dict(truncation=False, padding=False, verbose=False)
        return {'input_ids': [[101] + list(range(int(t))) + [102] for t in texts]}


def test_special_tokens_and_exact_limit():
    result = token_coverage(Tokenizer(), ['125', '126', '127'])
    assert [r['input_tokens'] for r in result] == [127, 128, 129]
    assert [r['truncated'] for r in result] == [False, False, True]


def test_empty_coverage_batch():
    assert token_coverage(Tokenizer(), []) == []
