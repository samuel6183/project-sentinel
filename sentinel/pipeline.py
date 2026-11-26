import argparse
import json
import sqlite3
from pathlib import Path
from sentinel.generate import generate
from sentinel.analysis import validate,providers,daily,reconciliation
from sentinel.model import evaluate

def run(n=100_000, seed=42, root=Path(".")):
    data=root/"data"; outputs=root/"outputs"
    data.mkdir(parents=True,exist_ok=True); outputs.mkdir(parents=True,exist_ok=True)
    df=generate(n,seed); validate(df)
    df.to_csv(data/"transactions.csv",index=False)
    with sqlite3.connect(data/"transactions.sqlite3") as db:
        df.to_sql("transactions",db,index=False,if_exists="replace")
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_transaction_id ON transactions(transaction_id)")
        db.execute("CREATE INDEX IF NOT EXISTS ix_provider_timestamp ON transactions(provider,timestamp)")
    providers(df).to_csv(outputs/"provider_metrics.csv",index=False)
    daily(df).to_csv(outputs/"daily_metrics.csv",index=False)
    reconciliation(df).to_csv(outputs/"reconciliation.csv",index=False)
    metrics=evaluate(df)
    (outputs/"model_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    return metrics

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--rows",type=int,default=100_000)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    print(json.dumps(run(a.rows,a.seed),indent=2))
