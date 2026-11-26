import streamlit as st
import plotly.express as px
from sentinel.generate import generate
from sentinel.analysis import providers,daily,reconciliation
from sentinel.model import evaluate

st.set_page_config(page_title="Project Sentinel",layout="wide")
st.title("Project Sentinel")
st.caption("Financial transaction intelligence | Fictional data only. No employer or customer records.")
@st.cache_data
def dataset():
    return generate()
df=dataset()
chosen=st.sidebar.multiselect("Provider",sorted(df.provider.unique()),default=sorted(df.provider.unique()))
directions=st.sidebar.multiselect("Direction",sorted(df.direction.unique()),default=sorted(df.direction.unique()))
f=df[df.provider.isin(chosen)&df.direction.isin(directions)]
if f.empty:
    st.warning("No transactions match these filters.")
    st.stop()
a,b,c,d=st.columns(4)
a.metric("Attempts",f"{len(f):,}")
b.metric("Approval rate",f"{f.status.eq('approved').mean():.1%}")
c.metric("Attempted volume (KES)",f"{f.amount.sum():,.0f}")
d.metric("Failure rate",f"{f.status.eq('failed').mean():.1%}")
p=providers(f)
st.subheader("Provider performance")
st.plotly_chart(px.bar(p,x="provider",y=["approval_rate","failure_rate","pending_rate"],barmode="group"),use_container_width=True)
st.dataframe(p,use_container_width=True)
st.subheader("Daily transaction attempts")
st.plotly_chart(px.line(daily(f),x="day",y="attempts"),use_container_width=True)
st.subheader("Settlement discrepancies")
r=reconciliation(f)
st.caption("Absolute discrepancies do not cancel out across transactions.")
st.metric("Absolute difference (KES)",f"{r.difference_kes.abs().sum():,.2f}")
st.dataframe(r,use_container_width=True)
st.download_button("Download filtered provider metrics",p.to_csv(index=False),"provider_metrics.csv","text/csv")
st.subheader("Synthetic-label model evaluation")
st.caption("Uses the full fictional dataset, independent of dashboard filters. Not validated for real-world fraud detection.")
@st.cache_data
def model_metrics():
    return evaluate(dataset())
if st.button("Run temporal holdout evaluation"):
    st.json(model_metrics())

st.divider()
st.header("Research reliability — V3")
st.caption("Independent simulation seeds, rolling temporal evaluation and hypothetical cost assumptions. Results do not establish real fraud performance.")
from pathlib import Path
import pandas as pd
summary_path=Path("outputs/v3_seed_summary.csv")
contrast_path=Path("outputs/v3_paired_cost_contrasts.csv")
summary_upload=st.file_uploader("Upload V3 seed summary CSV from the research artifact",type="csv",key="v3_summary")
contrast_upload=st.file_uploader("Upload V3 paired contrast CSV from the same artifact",type="csv",key="v3_contrast")
try:
    summary=pd.read_csv(summary_upload) if summary_upload is not None else pd.read_csv(summary_path) if summary_path.exists() else None
    contrast=pd.read_csv(contrast_upload) if contrast_upload is not None else pd.read_csv(contrast_path) if contrast_path.exists() else None
    if summary is None:
        st.info("No V3 results available locally. Download the completed reliability workflow artifact and upload the summary and paired-contrast CSVs.")
        st.markdown("[Reliability workflow](https://github.com/samuel6183/project-sentinel/actions/workflows/reliability.yml)")
    else:
        keys=["distribution","review_cost","miss_cost","capacity"]
        needed=set(keys+["model","independent_seeds","cost_per_1000","savings_per_1000","savings_per_1000_lo95","savings_per_1000_hi95","average_precision","recall","alert_rate"])
        if not needed.issubset(summary.columns):
            raise ValueError("Summary file is missing required V3 columns.")
        distribution=st.selectbox("Distribution scenario",sorted(summary.distribution.unique()))
        subset=summary[summary.distribution.eq(distribution)]
        options=list(subset[["review_cost","miss_cost","capacity"]].drop_duplicates().itertuples(index=False,name=None))
        scenario=st.selectbox("Cost and capacity scenario",options,format_func=lambda s:f"Review KES {s[0]} | Miss KES {s[1]} | Capacity {s[2]:.0%}")
        rc,mc,cap=scenario
        selected=subset[(subset.review_cost==rc)&(subset.miss_cost==mc)&(subset.capacity==cap)].copy()
        selected["upper_error"]=(selected.savings_per_1000_hi95-selected.savings_per_1000).clip(lower=0)
        selected["lower_error"]=(selected.savings_per_1000-selected.savings_per_1000_lo95).clip(lower=0)
        fig=px.bar(selected,x="model",y="savings_per_1000",error_y="upper_error",error_y_minus="lower_error",
                   labels={"savings_per_1000":"Hypothetical savings per 1,000 attempts (KES)"})
        fig.add_hline(y=0)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(selected[["model","independent_seeds","cost_per_1000","savings_per_1000","average_precision","recall","alert_rate"]],use_container_width=True)
        st.caption("Intervals summarize simulation-seed variability, not real-world uncertainty. Uploaded files are user-supplied and not authenticated by this dashboard.")
        if contrast is not None:
            required=set(keys+["mean_xgb_minus_rf_cost_per_1000","lo95","hi95","seeds"])
            if not required.issubset(contrast.columns):
                raise ValueError("Contrast file is missing required V3 columns.")
            match=contrast[(contrast.distribution==distribution)&(contrast.review_cost==rc)&(contrast.miss_cost==mc)&(contrast.capacity==cap)]
            if len(match)==1:
                row=match.iloc[0]
                st.metric("XGBoost minus Random Forest cost per 1,000 attempts",f"KES {row.mean_xgb_minus_rf_cost_per_1000:,.2f}")
                st.write(f"Approximate 95% interval: KES [{row.lo95:,.2f}, {row.hi95:,.2f}]")
                st.write("Interval crosses zero: inconclusive." if row.lo95<=0<=row.hi95 else "Interval favors XGBoost in this simulation." if row.hi95<0 else "Interval favors Random Forest in this simulation.")
except (ValueError,TypeError,KeyError,OSError) as exc:
    st.error(f"Unable to read reliability results: {exc}")
