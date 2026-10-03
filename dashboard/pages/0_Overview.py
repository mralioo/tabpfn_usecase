import streamlit as st

st.title("🚆 TabPFN-3.5 Railway Lab")
st.caption(
    "Started from [NextMove](https://github.com/mralioo/NextMove) — this repo goes deeper on "
    "just the TabPFN-3.5 piece, for the [TabPFN-3.5 hackathon](https://platform.priorlabs.ai/hackathon-3.5)."
)

st.markdown("""
Three railway datasets, used for two different purposes:

| Dataset | Role | Size | Real or simulated |
| --- | --- | --- | --- |
| 🚇 **Berlin U-Bahn** (Alstom / InnoTrans 2026) | Agent demo (MCP tool hand-off) + honest baseline check | 1.5M rows | Simulated flow, real geometry |
| 🚉 **Deutsche Bahn** (piebro/deutsche-bahn-data, HF) | TabPFN-3.5 vs. XGBoost, big real data | 1.5M rows / month | Real |
| 🇫🇮 **Finnish railway** (FI-TW-style, VR + FMI weather) | Biggest, multi-year — scale + drift exploration | ~500k rows / month × 96 months | Real |

**Where to go:**
- 📊 **Berlin — Explore** / 🇫🇮 **Finnish — Explore**: data exploration — what's actually in each
  dataset before any modeling decision gets made.
- 🔮 **Live Prediction**: ask the Berlin TabPFN-3.5 models a question directly.
- ⚖️ **Benchmark** pages: TabPFN-3.5 vs. a real baseline, with real numbers.

See `docs/MODULES.md` in the repo for the project's module layout and the plan for the next
phase — a scientific, same-methodology comparison of TabPFN-3.5 against XGBoost (and possibly
other models) across all three datasets.
""")
