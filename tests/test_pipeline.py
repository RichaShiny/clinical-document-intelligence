import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from clinical_intelligence.dataset import corpus, write_data
from clinical_intelligence.extract import extract, summarize
from clinical_intelligence.evaluate import counts, aggregate, bootstrap, paired_delta
from clinical_intelligence.llm import validate, extract_llm
from clinical_intelligence.pipeline import run


class ClinicalTests(unittest.TestCase):
    def test_gold_offsets_and_unique_ids(self):
        rows = corpus()
        self.assertEqual(len({r['id'] for r in rows}), len(rows))
        for row in rows:
            for e in row['gold']:
                self.assertEqual(row['text'][e['start']:e['end']], e['evidence'])

    def test_negation_and_clause_boundary(self):
        e = extract('No asthma but hypertension persists.')
        self.assertEqual([(x['value'],x['status']) for x in e], [('asthma','negated'),('hypertension','present')])

    def test_family_and_historical(self):
        self.assertEqual(extract('Mother has CHF.')[0]['status'], 'family')
        self.assertEqual(extract('Stopped metformin.')[0]['status'], 'historical')

    def test_abbreviation_and_boundaries(self):
        self.assertEqual(extract('HTN.')[0]['value'], 'hypertension')
        self.assertEqual(extract('asthmatic hypotension'), [])

    def test_summary_is_verbatim(self):
        text='No pneumonia. Ordered CXR. Follow-up in 1 week.'
        cards=summarize(text,extract(text))
        self.assertEqual(len(cards),3)
        for card in cards:
            self.assertEqual(text[card['start']:card['end']],card['text'])
        self.assertIn('negated', cards[0]['statuses'])

    def test_wrong_assertion_penalizes_both_precision_and_recall(self):
        gold=extract('No asthma.')
        prediction=extract('No asthma.',False)
        self.assertEqual(counts(gold,prediction),(0,1,1))

    def test_perfect_and_empty_metrics(self):
        self.assertEqual(aggregate([(1,0,0)])['f1'],1)
        self.assertEqual(aggregate([(0,0,0)])['f1'],0)
        self.assertEqual(bootstrap([(1,0,0)]),[1,1])

    def test_duplicates_not_double_counted(self):
        e=extract('Asthma.')
        self.assertEqual(counts(e,e+e),(1,0,0))

    def test_llm_rejects_fabricated_evidence_and_invalid_concept(self):
        raw={'entities':[dict(kind='condition',value='asthma',status='present',evidence='asthma'),
                         dict(kind='condition',value='cancer',status='present',evidence='cancer')]}
        accepted,rejected=validate(raw,'No asthma.')
        self.assertEqual(len(accepted),1)
        self.assertEqual(len(rejected),1)
        # Evidence matching alone deliberately does not establish assertion validity.
        self.assertEqual(accepted[0]['status'],'present')

    def test_llm_rejects_ambiguous_quote_and_bad_types(self):
        entity=dict(kind='condition',value='asthma',status='present',evidence='asthma')
        accepted,rejected=validate({'entities':[entity,{'kind':[]} ]},'asthma asthma')
        self.assertEqual(accepted,[])
        self.assertEqual(len(rejected),2)
        with self.assertRaises(ValueError): validate({'other':[]},'')

    def test_sentence_quote_alignment_preserves_assertion(self):
        raw={'entities':[dict(kind='condition',value='asthma',status='uncertain',evidence='No asthma.'),
            dict(kind='follow_up',value='in 7 days',status='planned',evidence='Follow-up in 7 days.')]}
        accepted,rejected=validate(raw,'No asthma. Follow-up in 7 days.')
        self.assertEqual(rejected,[])
        self.assertEqual(accepted[0]['evidence'],'asthma')
        self.assertEqual(accepted[0]['start'],3)
        self.assertEqual(accepted[0]['status'],'uncertain')
        self.assertEqual(accepted[1]['value'],'follow-up in 7 days')
        self.assertEqual(accepted[1]['evidence'],'Follow-up in 7 days')

    def test_alias_cannot_be_invented_from_other_evidence(self):
        raw={'entities':[dict(kind='condition',value='asthma',status='present',evidence='No hypertension.')]}
        self.assertEqual(validate(raw,'No hypertension.')[0],[])

    def test_ollama_protocol_with_mocked_transport(self):
        payload={'entities':[dict(kind='condition',value='asthma',status='negated',evidence='asthma')]}
        response=MagicMock()
        response.read.return_value=json.dumps({'message':{'content':json.dumps(payload)},'eval_count':9}).encode()
        response.__enter__.return_value=response
        with patch('clinical_intelligence.llm.urlopen',return_value=response) as transport:
            accepted,extra=extract_llm('No asthma.','test-model')
            request=transport.call_args.args[0]
            self.assertEqual(request.full_url,'http://127.0.0.1:11434/api/chat')
            self.assertFalse(json.loads(request.data)['stream'])
        self.assertEqual(accepted[0]['start'],3)
        self.assertEqual(extra['output_tokens'],9)

    def test_dataset_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            _,a=write_data(temp)
            _,b=write_data(temp)
            self.assertEqual(a,b)

    def test_end_to_end_exports_and_challenge_regression(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'dashboard').mkdir()
            report=run(root)
            self.assertEqual(report['dataset']['records'],166)
            self.assertGreater(report['systems']['contextual']['splits']['challenge']['f1'],.7)
            self.assertLess(report['systems']['contextual']['splits']['challenge']['f1'],.9)
            self.assertIn('not run',report['llm_status'])
            self.assertTrue((root/'dashboard'/'results.js').exists())
            self.assertAlmostEqual(report['paired_comparison']['delta_f1'],.45)

    def test_llm_coverage_and_matched_comparison(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'dashboard').mkdir()
            with patch('clinical_intelligence.analysis.capture_environment'), patch(
                'clinical_intelligence.llm.extract_llm',
                side_effect=lambda text,model: (extract(text),{'rejected':[]})), contextlib.redirect_stdout(io.StringIO()):
                report=run(root,'mock-model')
            result=report['systems']['ollama:mock-model']
            self.assertEqual(result['splits']['test']['n'],15)
            self.assertEqual(result['splits']['challenge']['n'],16)
            self.assertNotIn('dev',result['splits'])
            self.assertEqual(report['llm_comparison']['paired_test_delta']['delta_f1'],0)
            self.assertEqual(len(result['evaluated_note_ids']),31)

if __name__=='__main__': unittest.main()
