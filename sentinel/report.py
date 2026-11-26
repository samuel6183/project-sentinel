"""Produce a report from executed synthetic outputs, not hard-coded results."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def report(root=Path(".")):
    out=root/"outputs"
    df=pd.read_csv(root/"data/transactions.csv")
    p=pd.read_csv(out/"provider_metrics.csv")
    d=pd.read_csv(out/"daily_metrics.csv")
    m=json.loads((out/"model_metrics.json").read_text())
    p["approval_pct"]=p.approval_rate*100
    fig,ax=plt.subplots(figsize=(9,4))
    ax.bar(p.provider,p.approval_pct,color="#146553")
    ax.set(ylabel="Approval rate (%)",ylim=(0,100),title="Synthetic provider approval rates")
    fig.tight_layout();fig.savefig(out/"provider_approval.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(pd.to_datetime(d.day),d.attempts,color="#246284",linewidth=1)
    ax.set(ylabel="Attempts",title="Daily synthetic transaction volume")
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/"daily_activity.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(p.provider,p.absolute_discrepancy_kes,color="#b7791f")
    ax.set(ylabel="Absolute difference (KES)",title="Synthetic settlement discrepancies")
    fig.tight_layout();fig.savefig(out/"settlement.png",dpi=160);plt.close(fig)
    r=pd.read_csv(out/"reconciliation.csv")
    lines=["# Project Sentinel — executed synthetic analysis","",
       "All observations and labels are fictional. No real customer or employer data is represented.","",
       f"- Transaction attempts: {len(df):,}",
       f"- Gross attempted volume: KES {df.amount.sum():,.2f}",
       f"- Approval rate: {df.status.eq('approved').mean():.2%}",
       f"- Failure rate: {df.status.eq('failed').mean():.2%}",
       f"- Pending rate: {df.status.eq('pending').mean():.2%}",
       f"- Settlement exceptions: {len(r):,}",
       f"- Absolute settlement difference: KES {r.difference_kes.abs().sum():,.2f}",
       "", "## Provider results","",
       "| Provider | Attempts | Approval | Failure | P95 latency (s) | Absolute difference (KES) |",
       "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in p.itertuples():
        lines.append(f"| {row.provider} | {row.attempts:,} | {row.approval_rate:.2%} | {row.failure_rate:.2%} | {row.p95_seconds:.2f} | {row.absolute_discrepancy_kes:,.2f} |")
    lines += ["","![Provider approval](provider_approval.png)","","![Daily activity](daily_activity.png)","","![Settlement differences](settlement.png)","","## Model evaluation","",
      f"- Chronological holdout: {m['test_rows']:,} rows",
      f"- Synthetic label prevalence: {m['holdout_label_prevalence']:.2%}",
      f"- Random Forest ROC-AUC: {m['rf_roc_auc']:.4f}",
      f"- Random Forest average precision: {m['rf_average_precision']:.4f}",
      f"- Precision at fixed 0.5 threshold: {m['rf_precision_at_0_5']:.4f}",
      f"- Recall at fixed 0.5 threshold: {m['rf_recall_at_0_5']:.4f}",
      f"- F1 at fixed 0.5 threshold: {m['rf_f1_at_0_5']:.4f}",
      f"- Isolation Forest average precision: {m['isolation_forest_average_precision']:.4f}",
      "", "## Interpretation","",
      "Provider differences are consequences of simulated assumptions, not evidence about real providers. Average precision should be compared with holdout prevalence, and threshold metrics should not be treated as production performance. The model learns patterns in a synthetic label mechanism; real-world validation, calibration and cost-sensitive threshold selection would be required before deployment.","",
      "Reproduce with `python -m sentinel.pipeline --rows 100000 --seed 42` followed by `python -m sentinel.report`."]
    (out/"executive_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__":
    report()
