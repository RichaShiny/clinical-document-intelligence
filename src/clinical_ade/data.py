"""Pinned Hugging Face ADE ingestion with source-PMID recovery and grouped splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATASET='ade-benchmark-corpus/ade_corpus_v2'
REVISION='4ba01c71687dd7c996597042449448ea312126cf'
SOURCE_REVISION='2851af96e26050cf551a2e1f716ff861f7b1059b'
SPLITS={0:'train',1:'train',2:'train',3:'train',4:'train',5:'tune',6:'calibration',7:'policy',8:'test',9:'test'}


def normalize(text):
    return re.sub(r'\s+',' ',text).strip().casefold()


def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def download(url,path):
    if not path.exists():
        temp=path.with_suffix(path.suffix+'.part')
        urllib.request.urlretrieve(url,temp)
        temp.replace(path)
    return digest_file(path)


class UnionFind:
    def __init__(self):self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]]
            x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def spans_for(text,surface):
    return [(m.start(),m.end()) for m in re.finditer(re.escape(surface),text)]


def prepare(root):
    runtime=root/'runtime';runtime.mkdir(exist_ok=True)
    files={}
    for config in ['classification','drug_ade_relation']:
        files[config+'.parquet']=download(f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/Ade_corpus_v2_{config}/train-00000-of-00001.parquet',runtime/(config+'.parquet'))
    for f in ['DRUG-AE.rel','ADE-NEG.txt','README.txt']:
        files[f]=download(f'https://raw.githubusercontent.com/trunghlt/AdverseDrugReaction/{SOURCE_REVISION}/ADE-Corpus-V2/{f}',runtime/f)
    # Recover source document IDs without using labels or learned predictions for splitting.
    pmids=defaultdict(set)
    for line in (runtime/'DRUG-AE.rel').read_text().splitlines():
        parts=line.split('|')
        if len(parts)==8:pmids[normalize(parts[1])].add(parts[0])
    for line in (runtime/'ADE-NEG.txt').read_text().splitlines():
        parts=line.split(' NEG ',1)
        if len(parts)==2:pmids[normalize(parts[1])].add(parts[0])
    classification=pd.read_parquet(runtime/'classification.parquet')
    grouped=defaultdict(list)
    for item in classification.to_dict('records'):grouped[normalize(item['text'])].append(item)
    rows=[];conflicts=0;unmatched=0
    for norm,items in sorted(grouped.items()):
        if len({int(x['label']) for x in items})>1:
            conflicts+=1;continue
        ids=sorted(pmids[norm])
        if not ids:unmatched+=1;continue
        text=items[0]['text']
        rows.append(dict(id=hashlib.sha256(norm.encode()).hexdigest()[:20],text=text,label=int(items[0]['label']),pmids=ids,entities=[]))
    # Same normalized sentence in different papers connects those papers into one split group.
    uf=UnionFind()
    for row in rows:
        for pmid in row['pmids']:uf.union(row['pmids'][0],pmid)
    for row in rows:row['group']=uf.find(row['pmids'][0])
    by_norm={normalize(r['text']):r for r in rows}
    ambiguous=0;offset_disagreement=0
    relations=pd.read_parquet(runtime/'drug_ade_relation.parquet')
    for item in relations.to_dict('records'):
        row=by_norm.get(normalize(item['text']))
        if row is None:continue
        for kind,field in [('DRUG','drug'),('EFFECT','effect')]:
            locations=spans_for(row['text'],item[field])
            # Raw offsets are document-derived/inconsistent across distribution versions.
            # A unique literal surface is safely alignable; ambiguous repeats are excluded.
            if len(locations)!=1:ambiguous+=1;continue
            start,end=locations[0]
            raw_starts=item['indexes'][field]['start_char']
            if len(raw_starts)==0 or start!=int(raw_starts[0]):offset_disagreement+=1
            entity=dict(kind=kind,start=start,end=end)
            if entity not in row['entities']:row['entities'].append(entity)
    for row in rows:row['entities'].sort(key=lambda e:(e['start'],e['end'],e['kind']))
    y=np.array([r['label'] for r in rows]);groups=np.array([r['group'] for r in rows])
    splitter=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=42)
    for fold,(_,idx) in enumerate(splitter.split(np.zeros(len(rows)),y,groups)):
        for i in idx:rows[i]['split']=SPLITS[fold]
    assert_disjoint(rows)
    (runtime/'prepared.json').write_text(json.dumps(rows))
    manifest=dict(dataset=DATASET,revision=REVISION,source_id_revision=SOURCE_REVISION,sha256=files,
        raw_rows=len(classification),unique_normalized_sentences=len(grouped),retained_sentences=len(rows),
        conflicting_sentences_removed=conflicts,missing_pmid_removed=unmatched,
        duplicate_rows_removed=len(classification)-len(grouped),ambiguous_entity_alignments_excluded=ambiguous,
        document_vs_sentence_offset_disagreements=offset_disagreement,
        documents=len({p for r in rows for p in r['pmids']}),connected_document_groups=len(set(groups)),
        splits={s:dict(n=sum(r['split']==s for r in rows),positive=sum(r['label'] for r in rows if r['split']==s),
                      groups=len({r['group'] for r in rows if r['split']==s})) for s in sorted(set(SPLITS.values()))},
        split_policy='PMID groups connected through duplicate normalized text; stratified 10-fold seed42. Train50/tune10/calibration10/policy10/test20.',
        prepared_sha256=digest_file(runtime/'prepared.json'),license='Hugging Face card: unknown. Raw text and model artifacts are not redistributed in this repository.')
    (root/'reports').mkdir(exist_ok=True)
    (root/'reports/data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)
    return rows


def assert_disjoint(rows):
    seen_ids={};seen_text={}
    for row in rows:
        for pmid in row['pmids']:
            if pmid in seen_ids and seen_ids[pmid]!=row['split']:raise ValueError('Document leakage')
            seen_ids[pmid]=row['split']
        norm=normalize(row['text'])
        if norm in seen_text and seen_text[norm]!=row['split']:raise ValueError('Text leakage')
        seen_text[norm]=row['split']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());prepare(p.parse_args().root)

if __name__=='__main__':main()
