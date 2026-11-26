"""V4 fresh-seed confirmation, calibration and explanation diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, average_precision_score, log_loss
from xgboost import XGBClassifier
import shap
from sentinel.generate import generate
from sentinel.analysis import validate
from sentinel.advanced import features,choose_threshold,metrics
from sentinel.reliability import blocks,shifted_test,interval

SEEDS=tuple(range(2001,2021))
NAMES=("logistic_regression","xgboost")
REVIEW,MISS,CAP=25,1000,.05
# Fixed before execution, illustrative practical margin per 1,000 attempts.
MARGIN=250

def calibration_bins(y,p,edges):
    idx=np.clip(np.searchsorted(edges,p,side="right")-1,0,len(edges)-2)
    return [{"bin":i,"count":int((idx==i).sum()),"predicted_sum":float(p[idx==i].sum()),
             "observed_sum":int(y[idx==i].sum())} for i in range(len(edges)-1)]

def psi(reference,current):
    # Reference quantile bins fixed before inspecting the test period.
    edges=np.unique(np.quantile(reference,np.linspace(0,1,11)))
    if len(edges)<3:
        return 0.0 if np.array_equal(reference,current) else float("nan")
    edges[0],edges[-1]=-np.inf,np.inf
    a=np.histogram(reference,edges)[0].astype(float)+.5
    b=np.histogram(current,edges)[0].astype(float)+.5
    a/=a.sum();b/=b.sum()
    return float(np.sum((b-a)*np.log(b/a)))

def evaluate_seed(seed,n):
    df=generate(n,seed);validate(df)
    b=blocks(df)
    train=pd.concat(b[:7]);cal=b[7];val=b[8];test=b[9]
    assert train.timestamp.max()<cal.timestamp.min()<val.timestamp.min()<test.timestamp.min()
    xtr,xc,xv,xt=map(features,(train,cal,val,test))
    yt,yc,yv,ys=[p.synthetic_fraud_label.to_numpy() for p in (train,cal,val,test)]
    shifted=shifted_test(test,seed).synthetic_fraud_label.to_numpy()
    ratio=float((yt==0).sum()/(yt==1).sum())
    models={
        "logistic_regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42)),
        "xgboost":XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.85,colsample_bytree=.85,scale_pos_weight=ratio,eval_metric="logloss",random_state=42,n_jobs=2)}
    rows=[];bins=[];importance=[];drift=[]
    for name,model in models.items():
        model.fit(xtr,yt)
        def score(x):
            return model.predict_proba(x)[:,1]
        calibrator=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
        calibrator.fit(score(xc).reshape(-1,1),yc)
        def probability(x):
            return calibrator.predict_proba(score(x).reshape(-1,1))[:,1]
        pv,pt=probability(xv),probability(xt)
        chosen=choose_threshold(yv,pv,REVIEW,MISS,CAP)
        threshold=chosen["threshold"] if chosen["validation_alerts"] else 1.1
        edges=np.array([0,.005,.01,.015,.02,.03,.05,.10,1.000001])
        for condition,y in (("same_mechanism",ys),("label_shift_stress",shifted)):
            m=metrics(y,pt,threshold,REVIEW,MISS)
            baseline=float(y.sum()*MISS)
            rows.append({"seed":seed,"model":name,"condition":condition,"test_rows":len(y),
                         "prevalence":float(y.mean()),"threshold":threshold,"validation_cost":chosen["validation_cost"],
                         "cost_per_1000":1000*m["hypothetical_cost_kes"]/len(y),
                         "savings_per_1000":1000*(baseline-m["hypothetical_cost_kes"])/len(y),
                         "raw_brier":float(brier_score_loss(y,score(xt))),**m})
            bins.extend({"seed":seed,"model":name,"condition":condition,**z} for z in calibration_bins(y,pt,edges))
        # Fixed held-out sample; explanations describe raw model log-odds, not calibrated probabilities.
        sample=xt.sample(n=min(256,len(xt)),random_state=seed)
        if name=="xgboost":
            values=np.asarray(shap.TreeExplainer(model,feature_perturbation="tree_path_dependent",model_output="raw")(sample).values)
        else:
            scaler=model.named_steps["standardscaler"];lr=model.named_steps["logisticregression"]
            background=scaler.transform(xtr).mean(axis=0)
            values=(scaler.transform(sample)-background)*lr.coef_[0]
        for feature,v in zip(xt.columns,np.abs(values).mean(axis=0)):
            importance.append({"seed":seed,"model":name,"feature":feature,"mean_abs_raw_log_odds_attribution":float(v)})
        drift.append({"seed":seed,"model":name,"signal":"calibrated_score","psi":psi(pv,pt)})
    for feature in xt.columns:
        drift.append({"seed":seed,"model":"shared","signal":feature,"psi":psi(xv[feature].to_numpy(dtype=float),xt[feature].to_numpy(dtype=float))})
    return rows,bins,importance,drift

def run(n=100000,root=Path("."),seeds=SEEDS):
    out=root/"outputs";out.mkdir(parents=True,exist_ok=True)
    rows=[];bins=[];importance=[];drift=[]
    for seed in seeds:
        a,b,c,d=evaluate_seed(seed,n)
        rows.extend(a);bins.extend(b);importance.extend(c);drift.extend(d)
        print(f"Completed confirmation seed {seed}",flush=True)
    raw=pd.DataFrame(rows);cal=pd.DataFrame(bins);imp=pd.DataFrame(importance);dr=pd.DataFrame(drift)
    raw.to_csv(out/"v4_seed_results.csv",index=False)
    cal.to_csv(out/"v4_calibration_bins.csv",index=False)
    imp.to_csv(out/"v4_feature_attributions.csv",index=False)
    dr.to_csv(out/"v4_drift_diagnostics.csv",index=False)
    summary=[];contrasts=[]
    for condition,g in raw.groupby("condition"):
        for model,h in g.groupby("model"):
            row={"condition":condition,"model":model,"seeds":len(h)}
            for metric in ("cost_per_1000","savings_per_1000","average_precision","roc_auc","recall","precision","alert_rate","brier","raw_brier"):
                row[metric],row[metric+"_lo95"],row[metric+"_hi95"]=interval(h[metric])
            summary.append(row)
        pivot=g.pivot(index="seed",columns="model",values="cost_per_1000")
        mean,lo,hi=interval(pivot.xgboost-pivot.logistic_regression)
        contrasts.append({"condition":condition,"xgb_minus_logistic_cost_per_1000":mean,"lo95":lo,"hi95":hi,
                          "seeds":len(pivot),"within_practical_margin_95ci":bool(lo>-MARGIN and hi<MARGIN)})
    summary=pd.DataFrame(summary);contrasts=pd.DataFrame(contrasts)
    summary.to_csv(out/"v4_summary.csv",index=False);contrasts.to_csv(out/"v4_paired_contrasts.csv",index=False)
    # Pooled descriptive calibration plot; no transaction-level independence claim.
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot([0,.10],[0,.10],"--",color="grey",label="Ideal")
    for (condition,model),g in cal.groupby(["condition","model"]):
        agg=g.groupby("bin")[["count","predicted_sum","observed_sum"]].sum()
        agg=agg[agg["count"]>0]
        ax.plot(agg.predicted_sum/agg["count"],agg.observed_sum/agg["count"],marker="o",label=f"{condition}: {model}")
    ax.set(xlabel="Mean predicted probability",ylabel="Observed synthetic positive rate",title="Held-out calibration")
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/"v4_calibration.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,(name,g) in zip(axes,imp.groupby("model")):
        s=g.groupby("feature").mean_abs_raw_log_odds_attribution.agg(["mean","std"]).sort_values("mean")
        ax.barh(s.index,s["mean"],xerr=s["std"]);ax.set_title(name);ax.set_xlabel("Mean absolute raw log-odds attribution")
    fig.tight_layout();fig.savefig(out/"v4_attribution_stability.png",dpi=160);plt.close(fig)
    protocol={"seeds":list(seeds),"rows_per_seed":n,"primary":"XGBoost minus Logistic Regression cost per 1,000 on original mechanism",
              "practical_margin_kes":MARGIN,"costs":{"review":REVIEW,"miss":MISS,"validation_capacity":CAP},
              "limitations":"Fresh synthetic seeds only; stress is paired relabeling of the identical test features; attribution is descriptive, not causal; PSI is feature/score drift only."}
    (out/"v4_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    print("V4 SUMMARY");print(summary[["condition","model","seeds","cost_per_1000","savings_per_1000","average_precision","recall","brier","raw_brier"]].to_string(index=False))
    print("V4 PAIRED CONTRASTS");print(contrasts.to_string(index=False))
    print("V4 ATTRIBUTION STABILITY");print(imp.groupby(["model","feature"]).mean_abs_raw_log_odds_attribution.agg(["mean","std"]).to_string())
    return summary,contrasts

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--rows",type=int,default=100000)
    run(p.parse_args().rows)
