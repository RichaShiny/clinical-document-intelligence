"""Validated, source-linked inference with trusted artifact integrity checks."""
import hashlib
import json
from pathlib import Path
import re
import time
import joblib
import torch
from .data import digest_file
from .metrics import calibrated
from . import neural,entities


def sentences(text):
    # Literature sentence segmentation heuristic; abbreviation errors remain a limitation.
    for match in re.finditer(r'[^.!?\n]+(?:[.!?]+|$)',text):
        value=match.group();left=len(value)-len(value.lstrip());right=len(value.rstrip())
        if left<right:yield match.start()+left,match.start()+right,text[match.start()+left:match.start()+right]


class Engine:
    def __init__(self,root):
        self.root=Path(root);directory=self.root/'runtime/artifacts'
        self.manifest=json.loads((directory/'manifest.json').read_text())
        for name,digest in self.manifest['files'].items():
            if Path(name).name!=name:raise ValueError('Invalid artifact path')
            if digest_file(directory/name)!=digest:raise ValueError('Artifact integrity check failed: '+name)
        for name,digest in self.manifest.get('encoder_files',{}).items():
            if Path(name).name!=name or digest_file(self.root/'runtime/biomedbert'/name)!=digest:
                raise ValueError('Encoder artifact integrity check failed')
        self.name=self.manifest['selected_model'];self.version=digest_file(directory/'manifest.json')[:16]
        self.extractor_name=self.manifest.get('selected_extractor','crf')
        if self.extractor_name=='biomedbert_token':
            from .token_entities import TokenExtractor
            if 'entities_biomedbert_delta.pt' not in self.manifest['files']:
                raise ValueError('Biomedical entity checkpoint is not registered')
            required_encoder={'config.json','tokenizer_config.json','vocab.txt','pytorch_model.bin'}
            if not required_encoder.issubset(self.manifest.get('encoder_files',{})):
                raise ValueError('Biomedical entity encoder files are not registered')
            self.entity_model=TokenExtractor(self.root)
        elif self.extractor_name=='crf':
            self.crf=joblib.load(directory/'entities_crf.joblib')
        else:raise ValueError('Unsupported entity extractor')
        if self.name=='tfidf_logistic':
            bundle=joblib.load(directory/'tfidf_logistic.joblib');self.model=bundle['model']
        elif self.name=='pytorch_dan':
            bundle=joblib.load(directory/'pytorch_dan_policy.joblib')
            self.vocab=json.loads((directory/'vocabulary.json').read_text())
            self.model=neural.DeepAverage(len(self.vocab)+1)
            self.model.load_state_dict(torch.load(directory/'pytorch_dan.pt',map_location='cpu',weights_only=True));self.model.eval()
        elif self.name=='biomedbert_frozen':
            from transformers import AutoModel,AutoTokenizer
            bundle=joblib.load(directory/'biomedbert_head.joblib')
            location=str(self.root/'runtime/biomedbert')
            self.tokenizer=AutoTokenizer.from_pretrained(location,local_files_only=True)
            self.encoder=AutoModel.from_pretrained(location,local_files_only=True).eval()
            self.model=bundle['model'];self.scaler=bundle['scaler']
        else:raise ValueError('Unsupported serving model')
        self.cal=bundle['calibrator'];self.threshold=bundle['threshold']
        torch.set_num_threads(4)

    def analyze(self,text):
        start=time.perf_counter();spans=list(sentences(text))
        if len(spans)>200:raise ValueError('Maximum 200 sentences per request')
        if not spans:raise ValueError('No sentences found')
        texts=[s[2] for s in spans]
        if self.name=='tfidf_logistic':raw=self.model.predict_proba(texts)[:,1]
        elif self.name=='pytorch_dan':raw=neural.predict(self.model,neural.encode(texts,self.vocab))
        else:
            import numpy as np
            embeddings=[]
            with torch.inference_mode():
                for offset in range(0,len(texts),24):
                    inputs=self.tokenizer(texts[offset:offset+24],return_tensors='pt',padding=True,truncation=True,max_length=128)
                    hidden=self.encoder(**inputs).last_hidden_state
                    mask=inputs['attention_mask'].unsqueeze(-1)
                    embeddings.append(((hidden*mask).sum(1)/mask.sum(1)).numpy())
            raw=self.model.predict_proba(self.scaler.transform(np.concatenate(embeddings)))[:,1]
        probabilities=calibrated(self.cal,raw);out=[]
        candidates=[i for i,p in enumerate(probabilities) if p>=self.threshold]
        if self.extractor_name=='biomedbert_token':
            extracted_batch=self.entity_model.predict_batch([texts[i] for i in candidates])
        else:
            extracted_batch=[entities.predict(self.crf,texts[i]) for i in candidates]
        extracted_by_index=dict(zip(candidates,extracted_batch))
        for i,((left,right,sentence),probability) in enumerate(zip(spans,probabilities)):
            extracted=extracted_by_index.get(i,[])
            output=[dict(kind=e['kind'],text=e['text'],start=e['start']+left,end=e['end']+left) for e in extracted]
            out.append(dict(text=sentence,start=left,end=right,ade_probability=float(probability),
                action='review' if probability>=self.threshold else 'lower_priority',entities=output))
        review=[r for r in out if r['action']=='review']
        return dict(document_sha256=hashlib.sha256(text.encode()).hexdigest(),model=self.name,model_version=self.version,
            entity_extractor=self.extractor_name,threshold=float(self.threshold),sentences=out,evidence_summary=[dict(text=r['text'],start=r['start'],end=r['end']) for r in review],
            latency_ms=(time.perf_counter()-start)*1000,
            intended_use='Literature safety-review prioritization; no patient-specific recommendation or causal determination.')
