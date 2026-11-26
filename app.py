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
