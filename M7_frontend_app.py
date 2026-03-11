# Run with: streamlit run milestone7/app.py
# Requires FastAPI backend running on http://localhost:8000

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Activity 7.1 — Streamlit Selection ───────────────────────
# Streamlit chosen for: pure Python, rapid UI, built-in widgets,
# Plotly support, reactive updates, easy Streamlit Cloud deployment

st.set_page_config(page_title="Exam Anxiety Detector", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

# Custom styling
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .main-header {
    background: linear-gradient(135deg, #1B2A4A, #3D6BB5);
    padding: 2rem; border-radius: 16px; text-align: center;
    margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(29,53,87,0.3);
  }
  .main-header h1 { color: white; font-size: 2.2rem; margin: 0; }
  .main-header p  { color: rgba(255,255,255,0.75); margin-top: 0.5rem; }
  .result-low      { background:#eafaf1; border-left:5px solid #27ae60; border-radius:12px; padding:1.5rem; }
  .result-moderate { background:#fef9e7; border-left:5px solid #f39c12; border-radius:12px; padding:1.5rem; }
  .result-high     { background:#fdedec; border-left:5px solid #e74c3c; border-radius:12px; padding:1.5rem; }
  .tip-item { background:rgba(255,255,255,0.8); border-radius:8px; padding:0.55rem 1rem;
              margin:0.35rem 0; font-size:0.9rem; border-left:3px solid #3D6BB5; }
  .stButton > button { border-radius:10px!important; font-weight:700!important;
    background:linear-gradient(135deg,#1B2A4A,#3D6BB5)!important; color:white!important; border:none!important; }
</style>
""", unsafe_allow_html=True)

# ── Activity 7.2 — Connect Frontend to Backend API ───────────
API_URL = "https://ai-based-exam-anxiety-detector-production-1062.up.railway.app"

# Check if the FastAPI backend is reachable
def check_api_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=4).status_code == 200
    except Exception:
        return False

# Send student text to FastAPI and return the parsed response
def call_predict_api(text, student_id=""):
    try:
        payload = {"text": text}
        if student_id:
            payload["student_id"] = student_id
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Run: uvicorn backend.main:app --reload --port 8000")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# Session history stored across reruns
if "history" not in st.session_state:
    st.session_state.history = []

# Colour and emoji maps per anxiety level
COLOR = {"Low Anxiety": "#27AE60", "Moderate Anxiety": "#F39C12", "High Anxiety": "#E74C3C"}
EMOJI = {"Low Anxiety": "😊", "Moderate Anxiety": "😐", "High Anxiety": "😰"}
CSS   = {"Low Anxiety": "low", "Moderate Anxiety": "moderate", "High Anxiety": "high"}

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_ok = check_api_health()
    st.markdown(f"{'🟢' if api_ok else '🔴'} API: {'Connected' if api_ok else 'Disconnected'}")
    st.caption(f"`{API_URL}`")
    st.divider()

    student_id = st.text_input("Student ID (optional)", placeholder="STU_001")
    st.divider()

    st.markdown("#### 📊 Session Stats")
    if st.session_state.history:
        df_h = pd.DataFrame(st.session_state.history)
        st.metric("Total Analyses", len(df_h))
        for lbl, em in EMOJI.items():
            st.markdown(f"{em} **{lbl.split()[0]}**: {(df_h['label'] == lbl).sum()}")
    else:
        st.info("No analyses yet.")

    st.divider()
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.markdown("> ⚠️ *Supportive tool only — not a clinical diagnosis.*")

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎓 Exam Anxiety Detector</h1>
  <p>AI-powered mental wellness support · BERT NLP Model · Non-diagnostic</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Analyse", "📈 History", "ℹ️ About"])

# ── Tab 1 — Analyse ───────────────────────────────────────────
with tab1:
    col_in, col_out = st.columns([1.2, 0.9], gap="large")

    with col_in:
        st.markdown("### ✍️ Enter Your Pre-Exam Thoughts")

        # Sample texts for quick testing
        samples = {
            "— Select a sample —": "",
            "😊 Low Anxiety":      "I feel well prepared. I've covered all chapters and feel confident.",
            "😐 Moderate Anxiety": "I'm a bit nervous. Some topics are unclear and I keep second-guessing.",
            "😰 High Anxiety":     "I'm completely overwhelmed. I can't breathe and my mind goes blank.",
        }
        choice   = st.selectbox("Try a sample:", list(samples.keys()))
        prefill  = samples[choice]

        user_text = st.text_area("Your thoughts:", value=prefill, height=175,
                                  placeholder="e.g. I'm really nervous about tomorrow's exam...",
                                  label_visibility="collapsed")
        st.caption(f"Characters: {len(user_text)} / 1000")

        col_btn, col_clr = st.columns(2)
        with col_btn:
            go_clicked = st.button("🔍 Analyse Anxiety", use_container_width=True)
        with col_clr:
            if st.button("🔄 Clear", use_container_width=True):
                st.rerun()

    with col_out:
        st.markdown("### 📊 Result")

        if go_clicked:
            if len(user_text.strip()) < 5:
                st.warning("Please enter at least 5 characters.")
            else:
                with st.spinner("Analysing with BERT model…"):
                    response = call_predict_api(user_text, student_id)

                if response and response.get("success"):
                    res   = response["result"]
                    label = res["label"]

                    # Display result card
                    st.markdown(f"""
                    <div class="result-{CSS[label]}">
                        <h3 style="margin:0 0 4px 0">{res['emoji']} {label}</h3>
                        <p style="margin:0;font-size:0.9rem;color:#555">{res['message']}</p>
                        <p style="margin:6px 0 0;font-size:0.8rem;color:#888">
                            Confidence: <strong>{res['confidence']*100:.1f}%</strong>
                            &nbsp;|&nbsp; Inference: <strong>{res['inference_time_ms']:.0f}ms</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability bar chart
                    probs = res["probabilities"]
                    fig = go.Figure(go.Bar(
                        x=list(probs.values()), y=list(probs.keys()),
                        orientation="h",
                        marker_color=[COLOR[k] for k in probs],
                        text=[f"{v*100:.1f}%" for v in probs.values()],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="Class Probabilities",
                        xaxis=dict(range=[0, 1.2], tickformat=".0%"),
                        yaxis=dict(autorange="reversed"),
                        height=200, margin=dict(l=10, r=50, t=35, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Calming tips
                    st.markdown("#### 💡 Recommended Actions")
                    for tip in res["tips"]:
                        st.markdown(f'<div class="tip-item">• {tip}</div>', unsafe_allow_html=True)

                    # Save to session history
                    st.session_state.history.append({
                        "label": label, "level": res["level"],
                        "confidence": res["confidence"],
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "text": user_text[:60] + ("…" if len(user_text) > 60 else ""),
                    })
        else:
            # Placeholder before first analysis
            st.markdown("""
            <div style="background:#f8faff;border-radius:12px;padding:48px;text-align:center;
                        border:2px dashed #D6E4F7;">
                <div style="font-size:3rem">🧠</div>
                <p style="color:#aaa;margin-top:12px">
                    Enter text and click <strong>Analyse Anxiety</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)


# ── Tab 2 — History ───────────────────────────────────────────
with tab2:
    st.markdown("### 📈 Analysis History & Trends")

    if not st.session_state.history:
        st.info("No history yet — run some analyses first.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)

        # Summary metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(df_hist))
        c2.metric("Avg Confidence", f"{df_hist['confidence'].mean()*100:.1f}%")
        c3.metric("High Anxiety", int((df_hist["label"] == "High Anxiety").sum()))
        c4.metric("Most Common", df_hist["label"].mode()[0].split()[0])
        st.divider()

        # Trend scatter plot
        if len(df_hist) >= 2:
            df_hist["index"] = range(1, len(df_hist) + 1)
            fig = px.scatter(df_hist, x="index", y="level", color="label",
                             color_discrete_map=COLOR, title="Anxiety Level Trend",
                             labels={"index": "Analysis #", "level": "Anxiety Level"},
                             hover_data=["label", "confidence", "time"])
            fig.update_layout(
                yaxis=dict(tickvals=[1,2,3], ticktext=["Low","Moderate","High"], range=[0.5,3.5]),
                height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            fig.add_hline(y=2, line_dash="dot", line_color="#F39C12", opacity=0.4)
            st.plotly_chart(fig, use_container_width=True)

        # Distribution pie + history list side by side
        col_pie, col_list = st.columns([1, 1.4])
        with col_pie:
            counts  = df_hist["label"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                marker_colors=[COLOR[l] for l in counts.index], hole=0.4,
            ))
            fig_pie.update_layout(title="Distribution", height=280,
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_list:
            st.markdown("**Recent Entries**")
            for _, row in df_hist.iloc[::-1].head(8).iterrows():
                st.markdown(f"""
                <div style="background:white;border-radius:8px;padding:0.7rem 1rem;
                            margin:0.3rem 0;border:1px solid #eee;font-size:0.85rem;
                            display:flex;align-items:center;gap:10px;">
                    <span style="font-size:1.2rem">{EMOJI[row['label']]}</span>
                    <span style="flex:1;color:#555;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{row['text']}</span>
                    <span style="color:{COLOR[row['label']]};font-weight:700">{row['label'].split()[0]}</span>
                    <span style="color:#bbb">{row['time']}</span>
                </div>
                """, unsafe_allow_html=True)


# ── Tab 3 — About ─────────────────────────────────────────────
with tab3:
    st.markdown("### ℹ️ About This System")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        #### 🤖 How It Works
        Fine-tuned **BERT (bert-base-uncased)** analyses student text
        and classifies it into:
        - 😊 **Low Anxiety** — Calm and confident
        - 😐 **Moderate Anxiety** — Noticeable stress
        - 😰 **High Anxiety** — Overwhelming fear

        | Component | Technology |
        |-----------|------------|
        | NLP Model | BERT 110M params |
        | Training  | Google Colab GPU |
        | Backend   | FastAPI + PyTorch |
        | Frontend  | Streamlit + Plotly |
        """)

    with col_b:
        st.markdown("""
        #### 👥 Team
        | Member | Role |
        |--------|------|
        | Vedant Jadhav | ML & Backend |
        | Sameera Jadhav | Frontend & Training |
        | Kabir Jagtap | API & Evaluation |
        | Amar Jain | Data & Preprocessing |

        #### ⚠️ Disclaimer
        > This is a **non-diagnostic supportive tool**.
        > It does not replace professional mental health care.
        > For severe anxiety, please speak to a counselor.
        """)
