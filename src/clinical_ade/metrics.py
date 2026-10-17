"""Calibration, validation-only review policy, and document-cluster uncertainty."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_fscore_support,confusion_matrix,brier_score_loss,precision_recall_curve


def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


def calibrator(y,p):
    return LogisticRegression(C=1e6,random_state=42).fit(logit(p),y)


def calibrated(model,p):return model.predict_proba(logit(p))[:,1]


def choose_threshold(y,p,target=.90):
    precision,recall,thresholds=precision_recall_curve(y,p)
    possible=np.flatnonzero(recall[:-1]>=target)
    return float(thresholds[possible[-1]]) if len(possible) else 0.0


def reliability(y,p):
    bins=[];ece=0
    y=np.asarray(y);p=np.asarray(p)
    assignments=np.minimum((p*10).astype(int),9)
    for i in range(10):
        a=i/10
        mask=assignments==i
        if mask.sum():
            confidence=float(p[mask].mean());frequency=float(y[mask].mean())
            bins.append(dict(low=float(a),n=int(mask.sum()),probability=confidence,frequency=frequency))
            ece+=abs(confidence-frequency)*mask.mean()
    return float(ece),bins


def evaluate(y,p,threshold):
    y=np.asarray(y);p=np.asarray(p);pred=p>=threshold
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    ece,bins=reliability(y,p)
    return dict(n=len(y),positive=int(y.sum()),auprc=float(average_precision_score(y,p)),auroc=float(roc_auc_score(y,p)),
        precision=float(precision),recall=float(recall),f1=float(f1),threshold=float(threshold),
        review_fraction=float(pred.mean()),brier=float(brier_score_loss(y,p)),ece=ece,reliability=bins,
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())


def cluster_bootstrap(y,p,groups,threshold,samples=300):
    rng=np.random.default_rng(42);groups=np.asarray(groups);unique=np.unique(groups)
    indices={g:np.flatnonzero(groups==g) for g in unique};values=[]
    y=np.asarray(y);p=np.asarray(p)
    for _ in range(samples):
        idx=np.concatenate([indices[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(set(y[idx]))<2:continue
        pred=p[idx]>=threshold
        tp=np.sum(pred&(y[idx]==1));fn=np.sum(~pred&(y[idx]==1))
        values.append([average_precision_score(y[idx],p[idx]),tp/(tp+fn),pred.mean()])
    arr=np.array(values)
    return {k:np.quantile(arr[:,i],[.025,.975]).tolist() for i,k in enumerate(['auprc','recall','review_fraction'])}
