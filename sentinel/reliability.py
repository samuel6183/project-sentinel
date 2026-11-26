"""Pre-specified V3 replication protocol. Synthetic evidence only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sentinel.generate import generate
from sentinel.advanced import features, choose_threshold, metrics
from sentinel.analysis import validate

# Fixed before execution; seed 42 is development only.
SEEDS=(101,211,307,401,503,601,701,809,907,1009)
SCENARIOS=((25,1000,.05),(10,1000,.05),(50,1000,.05),(25,500,.05),
           (25,2000,.05),(25,1000,.02),(25,1000,.10))
MODELS=("logistic_regression","random_forest","xgboost","isolation_forest")
# Four consecutive held-out windows, after expanding fit and two separate tuning windows.
FOLDS=((4,5,6,7),(5,6,7,8),(6,7,8,9),(7,8,9,10))

def blocks(df):
    a=df.sort_values(["timestamp","transaction_id"]).copy()
    ts=pd.to_datetime(a.timestamp)
    cuts=[ts.iloc[int(len(a)*i/10)] for i in range(1,10)]
    if len(set(cuts))!=9:
        raise ValueError("Insufficient distinct timestamps")
    return [a.loc[(ts>=lo)&(ts<hi)] for lo,hi in zip([ts.min(),*cuts],[*cuts,ts.max()+pd.Timedelta(minutes=1)])]

def fit_scores(train,cal,val,test):
    xtr,xc,xv,xt=map(features,(train,cal,val,test))
    yt,yc,yv,ys=[p.synthetic_fraud_label.to_numpy() for p in (train,cal,val,test)]
    if any(np.unique(y).size!=2 for y in (yt,yc,yv,ys)):
        raise ValueError("Each partition must contain both classes")
    ratio=float((yt==0).sum()/(yt==1).sum())
    models={
        "logistic_regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42)),
        "random_forest":RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=15,class_weight="balanced",random_state=42,n_jobs=2),
        "xgboost":XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.85,colsample_bytree=.85,scale_pos_weight=ratio,eval_metric="logloss",random_state=42,n_jobs=2),
        "isolation_forest":IsolationForest(n_estimators=100,random_state=42,n_jobs=2)}
    scores={}
    for name,model in models.items():
        if name=="isolation_forest":
            model.fit(xtr)
            score=lambda x: -model.score_samples(x)
        else:
            model.fit(xtr,yt)
            score=lambda x: model.predict_proba(x)[:,1]
        calibrator=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
        calibrator.fit(score(xc).reshape(-1,1),yc)
        scores[name]=(calibrator.predict_proba(score(xv).reshape(-1,1))[:,1],
                      calibrator.predict_proba(score(xt).reshape(-1,1))[:,1])
    return yv,ys,scores,float(yt.mean())

def evaluate_fold(train,cal,val,test,seed,fold,scenario_name):
    yv,ys,scores,prevalence=fit_scores(train,cal,val,test)
    rows=[]
    for rc,mc,capacity in SCENARIOS:
        choices={name:choose_threshold(yv,pv,rc,mc,capacity) for name,(pv,pt) in scores.items()}
        champion=min(choices,key=lambda name:(choices[name]["validation_cost"],name))
        for name,(pv,pt) in scores.items():
            choice=choices[name]
            # A no-alert choice is a policy, not merely a threshold above a validation maximum.
            threshold=choice["threshold"] if choice["validation_alerts"] else 1.1
            m=metrics(ys,pt,threshold,rc,mc)
            rows.append({"seed":seed,"fold":fold,"distribution":scenario_name,"review_cost":rc,"miss_cost":mc,
                         "capacity":capacity,"model":name,"validation_champion":champion,
                         "validation_cost":choice["validation_cost"],"test_rows":len(ys),
                         "test_prevalence":float(ys.mean()),"threshold":threshold,
                         "baseline_cost":float(ys.sum()*mc),**m})
        m=metrics(ys,np.full(len(ys),prevalence),1.1,rc,mc)
        rows.append({"seed":seed,"fold":fold,"distribution":scenario_name,"review_cost":rc,"miss_cost":mc,
                     "capacity":capacity,"model":"no_alert_baseline","validation_champion":champion,
                     "validation_cost":float(yv.sum()*mc),"test_rows":len(ys),"test_prevalence":float(ys.mean()),
                     "threshold":1.1,"baseline_cost":float(ys.sum()*mc),**m})
    return rows

def shifted_test(test,seed):
    """Pre-specified alternative label mechanism, changing only held-out labels."""
    a=test.copy()
    rng=np.random.default_rng(seed+99991)
    logit=-4.9+.15*a.recent_tx_count+.40*(a.account_age_days<30)+.40*(a.attempt_count>3)+.35*(a.amount>15000)+.8*(a.direction=="withdrawal")+rng.normal(0,1.0,len(a))
    a["synthetic_fraud_label"]=rng.binomial(1,1/(1+np.exp(-logit)))
    return a

def interval(values):
    a=np.asarray(values,dtype=float)
    mean=float(a.mean())
    half=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else float("nan")
    return mean,mean-half,mean+half

def summarize(rows):
    df=pd.DataFrame(rows)
    df["cost_per_1000"]=1000*df.hypothetical_cost_kes/df.test_rows
    df["savings_per_1000"]=1000*(df.baseline_cost-df.hypothetical_cost_kes)/df.test_rows
    # Average folds within each seed first; seeds are the independent replication units.
    keys=["distribution","review_cost","miss_cost","capacity","model"]
    measures=["cost_per_1000","savings_per_1000","average_precision","roc_auc","brier","recall","precision","alert_rate"]
    per=df.groupby(keys+["seed"],as_index=False)[measures].mean()
    summary=[]
    for key,g in per.groupby(keys):
        row=dict(zip(keys,key));row["independent_seeds"]=len(g)
        for m in measures:
            row[m],row[m+"_lo95"],row[m+"_hi95"]=interval(g[m])
        summary.append(row)
    # Paired cost contrast: negative means XGBoost costs less than Random Forest.
    pivot=per.pivot(index=keys[:-1]+["seed"],columns="model",values="cost_per_1000").reset_index()
    pivot["xgb_minus_rf"]=pivot.xgboost-pivot.random_forest
    contrasts=[]
    for key,g in pivot.groupby(keys[:-1]):
        mean,lo,hi=interval(g.xgb_minus_rf)
        contrasts.append({**dict(zip(keys[:-1],key)),"mean_xgb_minus_rf_cost_per_1000":mean,"lo95":lo,"hi95":hi,"seeds":len(g)})
    return df,pd.DataFrame(summary),pd.DataFrame(contrasts)

def run(n=100000,root=Path("."),seeds=SEEDS):
    out=root/"outputs";out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        df=generate(n,seed);validate(df)
        b=blocks(df)
        for fold,(a,c,v,z) in enumerate(FOLDS,1):
            train=pd.concat(b[:a]);cal=pd.concat(b[a:c]);val=pd.concat(b[c:v]);test=pd.concat(b[v:z])
            assert train.timestamp.max()<cal.timestamp.min()<val.timestamp.min()<test.timestamp.min()
            rows.extend(evaluate_fold(train,cal,val,test,seed,fold,"same_mechanism"))
            # Stress the final rolling test only; no training or validation labels changed.
            if fold==len(FOLDS):
                rows.extend(evaluate_fold(train,cal,val,shifted_test(test,seed),seed,fold,"label_shift_stress"))
        print(f"Completed seed {seed}",flush=True)
    raw,summary,paired=summarize(rows)
    raw.to_csv(out/"v3_fold_results.csv",index=False)
    summary.to_csv(out/"v3_seed_summary.csv",index=False)
    paired.to_csv(out/"v3_paired_cost_contrasts.csv",index=False)
    base=summary[(summary.review_cost==25)&(summary.miss_cost==1000)&(summary.capacity==.05)]
    print("V3 primary results (mean across independent seeds; cost per 1000 attempts):")
    print(base[["distribution","model","independent_seeds","cost_per_1000","savings_per_1000","savings_per_1000_lo95","savings_per_1000_hi95","average_precision","recall","alert_rate"]].to_string(index=False))
    print("V3 paired XGBoost minus Random Forest cost contrasts:")
    print(paired.to_string(index=False))
    protocol={"seeds":list(seeds),"rows_per_seed":n,"folds":FOLDS,"scenarios":SCENARIOS,
              "primary_contrast":"XGBoost minus Random Forest mean cost per 1000 attempts",
              "intervals":"Approximate two-sided 95% Student t intervals across independent seed means; exploratory scenario intervals not multiplicity adjusted.",
              "limitations":"All data synthetic; generator mechanisms fixed; stress is one alternative label mechanism; no external validation."}
    (out/"v3_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    return raw,summary,paired

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--rows",type=int,default=100000)
    a=p.parse_args()
    run(a.rows)
