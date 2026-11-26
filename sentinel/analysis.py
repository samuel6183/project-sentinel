"""Operational metrics; monetary amounts are gross synthetic KES."""
import pandas as pd
import numpy as np

def validate(df):
    required={"transaction_id","timestamp","provider","direction","method","currency","amount","status",
    "processing_seconds","expected_settlement","actual_settlement","synthetic_fraud_label","attempt_count","account_age_days","recent_tx_count"}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError("Missing columns or empty data")
    if df[list(required)].isna().any().any() or not df.transaction_id.is_unique:
        raise ValueError("Missing values or duplicate IDs")
    if not df.status.isin(["approved","failed","pending"]).all() or not df.currency.eq("KES").all():
        raise ValueError("Invalid status or currency")
    numeric=["amount","processing_seconds","expected_settlement","actual_settlement","attempt_count","account_age_days","recent_tx_count"]
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all() or (df[numeric]<0).any().any():
        raise ValueError("Invalid numeric values")
    if not df.synthetic_fraud_label.isin([0,1]).all():
        raise ValueError("Invalid synthetic label")

def providers(df):
    validate(df)
    a=df.assign(approved=df.status.eq("approved"),failed=df.status.eq("failed"),pending=df.status.eq("pending"),
        discrepancy=(df.actual_settlement-df.expected_settlement).abs())
    return a.groupby("provider",as_index=False).agg(attempts=("transaction_id","size"),
      attempted_volume_kes=("amount","sum"),approval_rate=("approved","mean"),failure_rate=("failed","mean"),
      pending_rate=("pending","mean"),median_seconds=("processing_seconds","median"),
      p95_seconds=("processing_seconds",lambda s:s.quantile(.95)),
      absolute_discrepancy_kes=("discrepancy","sum"),synthetic_label_rate=("synthetic_fraud_label","mean"))

def daily(df):
    validate(df)
    a=df.assign(day=pd.to_datetime(df.timestamp).dt.floor("D"),approved=df.status.eq("approved"))
    return a.groupby("day",as_index=False).agg(attempts=("transaction_id","size"),approval_rate=("approved","mean"),attempted_volume_kes=("amount","sum"))

def reconciliation(df):
    validate(df)
    a=df.assign(difference_kes=(df.actual_settlement-df.expected_settlement).round(2))
    return a.loc[a.difference_kes.ne(0),["transaction_id","timestamp","provider","direction","expected_settlement","actual_settlement","difference_kes"]]
