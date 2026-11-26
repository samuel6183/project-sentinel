"""Chronological model comparison; all labels and costs are hypothetical."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix, log_loss)
from xgboost import XGBClassifier
from sentinel.generate import generate
from sentinel.analysis import validate

FEATURES=["amount","attempt_count","account_age_days","recent_tx_count"]

def split(df):
    # Keep entire timestamp groups together to prevent boundary overlap.
    a=df.sort_values(["timestamp","transaction_id"]).copy()
    times=pd.to_datetime(a.timestamp)
    boundaries=[times.iloc[int(len(a)*p)] for p in (.60,.70,.80)]
    parts=[a.loc[times<boundaries[0]],a.loc[(times>=boundaries[0])&(times<boundaries[1])],
           a.loc[(times>=boundaries[1])&(times<boundaries[2])],a.loc[times>=boundaries[2]]]
    if any(len(p)<20 or p.synthetic_fraud_label.nunique()!=2 for p in parts):
        raise ValueError("Each chronological partition needs both classes and at least 20 rows")
    return parts

def features(df):
    x=df[FEATURES].astype(float).copy()
    x["log_amount"]=np.log1p(x.amount)
    x["new_account"]=x.account_age_days.lt(30).astype(float)
    x["retry_intensity"]=x.attempt_count*x.recent_tx_count
    return x

def choose_threshold(y,p,review_cost=25,miss_cost=1000,max_alert_rate=.05):
    """Illustrative loss: review cost per alert plus missed-positive cost.
    Choose on validation only; budget is a validation constraint, not a guaranteed test limit."""
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if len(y)==0 or len(y)!=len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid scores")
    if review_cost<0 or miss_cost<0 or not 0<=max_alert_rate<=1:
        raise ValueError("Invalid cost assumptions")
    order=np.argsort(-p,kind="stable")
    scores=p[order];labels=y[order]
    # Consider every unique score boundary; never split tied scores.
    k=np.r_[0,np.flatnonzero(scores[:-1]!=scores[1:])+1,len(y)]
    k=k[k<=int(len(y)*max_alert_rate)]
    tp=np.r_[0,np.cumsum(labels)][k]
    costs=k*review_cost+(y.sum()-tp)*miss_cost
    i=int(np.argmin(costs));count=int(k[i])
    threshold=float(np.nextafter(scores[0],np.inf) if count==0 else scores[count-1])
    return {"threshold":threshold,"validation_alerts":count,"validation_cost":float(costs[i])}

def metrics(y,p,threshold,review_cost,miss_cost):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"roc_auc":float(roc_auc_score(y,p)),
      "average_precision":float(average_precision_score(y,p)),
      "brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),
      "precision":float(precision_score(y,pred,zero_division=0)),
      "recall":float(recall_score(y,pred,zero_division=0)),
      "f1":float(f1_score(y,pred,zero_division=0)),
      "alerts":int(pred.sum()),"alert_rate":float(pred.mean()),
      "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
      "hypothetical_cost_kes":float(pred.sum()*review_cost+fn*miss_cost)}

def compare(df,review_cost=25,miss_cost=1000,max_alert_rate=.05):
    validate(df)
    train,cal,val,test=split(df)
    xtr,xc,xv,xt=map(features,(train,cal,val,test))
    yt,yc,yv,ys=[p.synthetic_fraud_label.to_numpy() for p in (train,cal,val,test)]
    ratio=float((yt==0).sum()/(yt==1).sum())
    models={
      "logistic_regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42)),
      "random_forest":RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=15,class_weight="balanced",random_state=42,n_jobs=2),
      "xgboost":XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.85,colsample_bytree=.85,scale_pos_weight=ratio,eval_metric="logloss",random_state=42,n_jobs=2),
      "isolation_forest":IsolationForest(n_estimators=100,random_state=42,n_jobs=2)}
    staged=[]
    for name,model in models.items():
        if name=="isolation_forest":
            model.fit(xtr)
            score=lambda x: -model.score_samples(x)
        else:
            model.fit(xtr,yt)
            score=lambda x: model.predict_proba(x)[:,1]
        # Platt-style logistic calibration on disjoint chronological data.
        calibrator=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
        calibrator.fit(score(xc).reshape(-1,1),yc)
        pv=calibrator.predict_proba(score(xv).reshape(-1,1))[:,1]
        pt=calibrator.predict_proba(score(xt).reshape(-1,1))[:,1]
        selected=choose_threshold(yv,pv,review_cost,miss_cost,max_alert_rate)
        staged.append((name,selected,metrics(yv,pv,selected["threshold"],review_cost,miss_cost),pt))
    # Freeze champion selection before reading any final test metrics.
    champion=min(staged,key=lambda item:(item[2]["hypothetical_cost_kes"],-item[2]["average_precision"]))[0]
    rows=[]
    scenarios=[]
    for name,selected,vm,pt in staged:
        rows.append({"model":name,**selected,"validation_average_precision":vm["average_precision"],
                     **metrics(ys,pt,selected["threshold"],review_cost,miss_cost)})
    # Baseline: no review alerts. Probability baseline is training prevalence.
    baseline=np.full(len(ys),yt.mean())
    rows.append({"model":"no_alert_baseline","threshold":1.1,"validation_alerts":0,
                 "validation_cost":float(yv.sum()*miss_cost),"validation_average_precision":float(yv.mean()),
                 **metrics(ys,baseline,1.1,review_cost,miss_cost)})
    # Model fixed to validation-selected champion; sensitivity thresholds reselected on validation.
    # Report separate assumed costs; never optimize on final test.
    info={"champion_selected_on_validation":champion,"review_cost_kes":review_cost,
          "miss_cost_kes":miss_cost,"max_validation_alert_rate":max_alert_rate,
          "test_prevalence":float(ys.mean()),
          "partitions":[{"role":role,"rows":len(p),"start":str(p.timestamp.min()),"end":str(p.timestamp.max()),
                         "positive_rate":float(p.synthetic_fraud_label.mean())}
                        for role,p in zip(("train","calibration","validation","test"),(train,cal,val,test))],
          "features":list(xtr.columns),"limitations":"Synthetic labels, hypothetical costs, single temporal holdout; not deployment evidence."}
    return pd.DataFrame(rows),info

def run(n=100000,seed=42,root=Path(".")):
    out=root/"outputs";out.mkdir(parents=True,exist_ok=True)
    table,info=compare(generate(n,seed))
    table.to_csv(out/"advanced_model_comparison.csv",index=False)
    (out/"advanced_model_protocol.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info,indent=2))
    print(table.to_string(index=False))
    return table,info

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--rows",type=int,default=100000)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    run(a.rows,a.seed)
