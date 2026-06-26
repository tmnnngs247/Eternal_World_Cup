import streamlit as st


def inject_css() -> None:
    st.markdown("""
    <style>
    :root {
        --ewc-bg: #0B1020;
        --ewc-panel: rgba(21, 31, 52, 0.96);
        --ewc-panel-2: rgba(32, 45, 73, 0.96);
        --ewc-text: #F8FAFC;
        --ewc-muted: #D6E0EE;
        --ewc-soft: #AFC1D4;
        --ewc-border: rgba(203, 213, 225, 0.24);
        --ewc-accent: #7DD3FC;
        --ewc-coral: #FB7185;
        --ewc-green: #86EFAC;
        --ewc-purple: #C084FC;
    }
    .stApp {
        background: radial-gradient(circle at 12% 0%, rgba(125,211,252,.20), transparent 28rem),
                    radial-gradient(circle at 88% 4%, rgba(192,132,252,.18), transparent 30rem),
                    linear-gradient(180deg, #0B1020 0%, #090D18 100%);
        color: var(--ewc-text);
    }
    [data-testid="stHeader"] { background: rgba(11,16,32,.72); backdrop-filter: blur(10px); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(14,24,44,.98), rgba(8,18,38,.98)); border-right:1px solid var(--ewc-border); }
    [data-testid="stSidebar"] * { color: var(--ewc-muted) !important; }
    .block-container { padding-top: 3.2rem; max-width: 1280px; }
    h1,h2,h3 { color: var(--ewc-text); letter-spacing: -.035em; }
    p, li, label, span, div { color: inherit; }
    .ewc-hero { padding: 3.4rem 3rem; border:1px solid var(--ewc-border); border-radius: 2rem;
        background: linear-gradient(135deg, rgba(35,58,87,.92), rgba(41,33,64,.92)); box-shadow: 0 24px 80px rgba(0,0,0,.28); }
    .ewc-hero h1 { font-size: clamp(3rem, 6vw, 5rem); margin:0 0 1.3rem 0; }
    .ewc-hero p { color: var(--ewc-muted); font-size: 1.18rem; line-height:1.65; max-width: 960px; }
    .pill { display:inline-flex; padding:.55rem 1rem; border:1px solid var(--ewc-border); border-radius:999px; background:rgba(255,255,255,.055); margin:.3rem .45rem .1rem 0; color:#F8FAFC; }
    .metric-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:1rem; margin:1.25rem 0; }
    .metric-card { border:1px solid var(--ewc-border); border-radius:1.2rem; background: rgba(15,23,42,.86); padding:1.35rem 1.5rem; }
    .metric-label { color: var(--ewc-soft); font-size:.95rem; margin-bottom:.45rem; }
    .metric-value { color: var(--ewc-text); font-size:2.1rem; font-weight:800; line-height:1; }
    .metric-note { color: var(--ewc-soft); margin-top:.55rem; font-size:.92rem; }
    .info-callout { border-left:4px solid var(--ewc-accent); padding:1.15rem 1.3rem; border-radius:1rem; background:rgba(125,211,252,.08); color:#F8FAFC; line-height:1.6; }
    .player-card { border:1px solid var(--ewc-border); border-radius:1.25rem; background:linear-gradient(135deg, rgba(17,28,48,.98), rgba(23,34,58,.92)); padding:1.15rem; margin:.75rem 0; }
    .player-title { font-size:1.25rem; font-weight:800; color:#F8FAFC; }
    .player-meta { color:var(--ewc-soft); font-size:.95rem; margin-top:.15rem; }
    .score-chip { display:inline-flex; align-items:center; justify-content:center; min-width:72px; padding:.45rem .7rem; border-radius:999px; background:rgba(125,211,252,.14); color:#E0F2FE; font-weight:800; }
    .section-card { border:1px solid var(--ewc-border); border-radius:1.5rem; padding:1.6rem; background:rgba(15,23,42,.72); margin-top:1rem; }
    @media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } .ewc-hero { padding:2rem; } }
    @media (max-width: 560px) { .metric-grid { grid-template-columns: 1fr; } }
    </style>
    """, unsafe_allow_html=True)

    /* Improve Streamlit dropdown/select contrast */
div[data-baseweb="select"] > div {
  background-color: #f3f6fb !important;
  color: #111827 !important;
}

div[data-baseweb="select"] span {
  color: #111827 !important;
}

div[data-baseweb="popover"] {
  color: #111827 !important;
}

ul[role="listbox"] li {
  color: #111827 !important;
}
