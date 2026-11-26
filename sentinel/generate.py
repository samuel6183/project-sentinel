"""Reproducible fictional payment attempts; not derived from any employer."""
import numpy as np
import pandas as pd

def generate(n=100_000, seed=42):
    if n < 1:
        raise ValueError("n must be positive")
    rng=np.random.default_rng(seed)
    providers=rng.choice(["PSP-A","PSP-B","PSP-C","PSP-D"],n,p=[.35,.30,.22,.13])
    direction=rng.choice(["deposit","withdrawal"],n,p=[.75,.25])
    method=rng.choice(["mobile_money","card","bank_transfer"],n,p=[.55,.30,.15])
    timestamps=pd.Timestamp("2025-01-01")+pd.to_timedelta(rng.integers(0,365*24*60,n),unit="min")
    amount=np.round(np.clip(rng.lognormal(7.5,1.1,n),50,100000),2)
    attempts=rng.poisson(1.3,n)+1
    account_age=rng.integers(1,1500,n)
    velocity=rng.poisson(2.5,n)
    provider_effect=pd.Series(providers).map({"PSP-A":.045,"PSP-B":.07,"PSP-C":.095,"PSP-D":.12}).to_numpy()
    fail_prob=np.clip(provider_effect+.025*(direction=="withdrawal")+.03*(attempts>3),0,.8)
    u=rng.random(n)
    status=np.where(u<fail_prob,"failed",np.where(u<fail_prob+.035,"pending","approved"))
    latency=np.round(rng.lognormal(2.5,.7,n)*(1+provider_effect*5),2)
    # Hidden simulation driver adds uncertainty; label is fictional, not a real fraud finding.
    logit=-5.4+.30*velocity+.85*(account_age<30)+.8*(attempts>3)+.7*(amount>15000)+rng.normal(0,.8,n)
    fraud=rng.binomial(1,1/(1+np.exp(-logit)))
    expected=np.where(status=="approved",amount,0.0)
    delta=np.where((status=="approved")&(rng.random(n)<.008),rng.choice([-100.,-20.,20.,100.],n),0.)
    actual=np.maximum(0,expected+delta)
    df=pd.DataFrame({"transaction_id":[f"SIM-{i:07d}" for i in range(n)],"timestamp":timestamps,
      "provider":providers,"direction":direction,"method":method,"currency":"KES","amount":amount,
      "attempt_count":attempts,"account_age_days":account_age,"recent_tx_count":velocity,
      "status":status,"processing_seconds":latency,"expected_settlement":expected,
      "actual_settlement":actual,"synthetic_fraud_label":fraud})
    return df.sort_values(["timestamp","transaction_id"]).reset_index(drop=True)
