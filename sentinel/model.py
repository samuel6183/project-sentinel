"""Temporal holdout evaluation on synthetic labels; not production fraud detection."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
FEATURES=["amount","attempt_count","account_age_days","recent_tx_count"]

def evaluate(df):
    a=df.sort_values(["timestamp","transaction_id"])
    cut=int(len(a)*.8)
    train,test=a.iloc[:cut],a.iloc[cut:]
    x,y=train[FEATURES],train.synthetic_fraud_label
    xt,yt=test[FEATURES],test.synthetic_fraud_label
    if len(train)<20 or y.nunique()!=2 or yt.nunique()!=2:
        raise ValueError("Both classes required in train and test")
    rf=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=15,class_weight="balanced",random_state=42,n_jobs=-1)
    rf.fit(x,y)
    # Threshold selected in advance, not optimized against the holdout.
    proba=rf.predict_proba(xt)[:,1]
    pred=proba>=.5
    iso=IsolationForest(n_estimators=100,contamination=.02,random_state=42,n_jobs=-1)
    iso.fit(x)
    anomaly=-iso.score_samples(xt)
    return {"train_rows":len(train),"test_rows":len(test),"holdout_label_prevalence":float(yt.mean()),
      "rf_roc_auc":float(roc_auc_score(yt,proba)),"rf_average_precision":float(average_precision_score(yt,proba)),
      "rf_precision_at_0_5":float(precision_score(yt,pred,zero_division=0)),
      "rf_recall_at_0_5":float(recall_score(yt,pred,zero_division=0)),
      "rf_f1_at_0_5":float(f1_score(yt,pred,zero_division=0)),
      "rf_confusion_matrix":confusion_matrix(yt,pred,labels=[0,1]).tolist(),
      "isolation_forest_average_precision":float(average_precision_score(yt,anomaly))}
