"""Trainable PyTorch Deep Averaging Network; no pretrained weights."""
from collections import Counter
import re
import numpy as np
import torch
from torch import nn


def tokenize(text):return re.findall(r"\b[\w'-]+\b",text.lower())


def vocabulary(texts,max_words=30000):
    counts=Counter(t for text in texts for t in tokenize(text))
    return {word:i+1 for i,(word,n) in enumerate(counts.most_common(max_words)) if n>=2}


def encode(texts,vocab):return [[vocab.get(w,0) for w in tokenize(text)][:256] or [0] for text in texts]


def batch_tokens(rows):
    lengths=[len(r) for r in rows]
    return (torch.tensor([w for r in rows for w in r],dtype=torch.long),
            torch.tensor([0]+list(np.cumsum(lengths)[:-1]),dtype=torch.long))


class DeepAverage(nn.Module):
    def __init__(self,size):
        super().__init__();self.embedding=nn.EmbeddingBag(size,96,mode='mean')
        self.head=nn.Sequential(nn.Linear(96,64),nn.ReLU(),nn.Dropout(.25),nn.Linear(64,1))
    def forward(self,tokens,offsets):return self.head(self.embedding(tokens,offsets)).squeeze(-1)


def predict(model,rows,batch_size=256):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):out.extend(torch.sigmoid(model(*batch_tokens(rows[start:start+batch_size]))).tolist())
    return np.array(out)
