"""Conditional random field for ADE-associated DRUG/EFFECT spans, not causality."""
import re
import sklearn_crfsuite


def tokens(text):return [(m.group(),m.start(),m.end()) for m in re.finditer(r'\w+(?:[-\']\w+)*|[^\w\s]',text)]


def features(seq,i,lexicon=None):
    word=seq[i][0]
    f={'bias':1.,'word':word.lower(),'suffix3':word[-3:].lower(),'suffix4':word[-4:].lower(),
       'upper':word.isupper(),'title':word.istitle(),'digit':word.isdigit()}
    if lexicon:
        f['drug_lexicon']=word.lower() in lexicon['DRUG']
        f['effect_lexicon']=word.lower() in lexicon['EFFECT']
    if i>=2:f['prev2']=seq[i-2][0].lower()
    if i+2<len(seq):f['next2']=seq[i+2][0].lower()
    if i:f.update(prev=seq[i-1][0].lower(),prev_title=seq[i-1][0].istitle())
    else:f['BOS']=True
    if i+1<len(seq):f['next']=seq[i+1][0].lower()
    else:f['EOS']=True
    return f


def sample(row,lexicon=None):
    seq=tokens(row['text']);labels=['O']*len(seq);overlap=0
    for entity in row['entities']:
        idx=[i for i,(_,start,end) in enumerate(seq) if start>=entity['start'] and end<=entity['end']]
        if not idx or seq[idx[0]][1]!=entity['start'] or seq[idx[-1]][2]!=entity['end']:continue
        if any(labels[i]!='O' for i in idx):overlap+=1;continue
        for j,i in enumerate(idx):labels[i]=('B-' if j==0 else 'I-')+entity['kind']
    return [features(seq,i,lexicon) for i in range(len(seq))],labels,overlap


def decode(text,labels):
    seq=tokens(text);spans=[];current=None
    for (word,start,end),tag in zip(seq,labels):
        if tag=='O':
            current=None;continue
        prefix,kind=tag.split('-',1)
        if prefix=='B' or current is None or current['kind']!=kind:
            current=dict(kind=kind,start=start,end=end);spans.append(current)
        else:current['end']=end
    return [dict(e,text=text[e['start']:e['end']]) for e in spans]


def train(rows):
    lexicon={k:set() for k in ['DRUG','EFFECT']}
    for row in rows:
        for e in row['entities']:
            lexicon[e['kind']].update(w.lower() for w,_,_ in tokens(row['text'][e['start']:e['end']]))
    samples=[sample(r,lexicon) for r in rows]
    model=sklearn_crfsuite.CRF(algorithm='lbfgs',c1=.1,c2=.1,max_iterations=80,all_possible_transitions=True)
    model.fit([s[0] for s in samples],[s[1] for s in samples])
    model.ade_lexicon=lexicon
    return model,sum(s[2] for s in samples)


def predict(model,text):
    seq=tokens(text)
    return decode(text,model.predict_single([features(seq,i,getattr(model,'ade_lexicon',None)) for i in range(len(seq))])) if seq else []


def evaluate(model,rows,gates=None):
    tp=fp=fn=0;by_kind={}
    for kind in ['DRUG','EFFECT']:
        a=b=c=0
        for row in rows:
            gold={(e['start'],e['end']) for e in row['entities'] if e['kind']==kind}
            pred={(e['start'],e['end']) for e in (predict(model,row['text']) if gates is None or gates.get(row['id'],False) else []) if e['kind']==kind}
            a+=len(gold&pred);b+=len(pred-gold);c+=len(gold-pred)
        tp+=a;fp+=b;fn+=c
        by_kind[kind]=dict(tp=a,fp=b,fn=c,f1=2*a/(2*a+b+c) if 2*a+b+c else 0.)
    return dict(tp=tp,fp=fp,fn=fn,micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,by_kind=by_kind,
                scope='Exact ADE-associated spans in all test sentences; relation causality and arbitrary drug mentions are not annotated exhaustively.')
