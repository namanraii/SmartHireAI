"""
app.py — SmartHire AI Streamlit Application
"""
import os
# MAC SPECIFIC FIX: Prevent Segmentation Faults when loading deep learning models in Streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# Local modules
from utils.resume_parser import parse_resume, extract_sections
from utils.preprocessor import preprocess, extract_noun_phrases
from utils.ner_extractor import extract_all_entities
from utils.skill_gap import compute_skill_gap
from utils.similarity_engine import combined_similarity
from utils.experience_scorer import compute_weighted_score
from utils.bias_detector import detect_bias_flags
from ml.predict import predict_all_models, get_feature_vector, get_model_metrics
from ml.shap_explainer import compute_shap_values, plot_shap_bar, get_top_factors
from features.recommender import generate_learning_path
from features.interview_qgen import generate_questions
from features.ranker import rank_candidates
from features.report_generator import generate_pdf_report


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartHire AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theming & CSS Configuration ──────────────────────────────────────────────
def inject_custom_css(is_dark=True):
    # Consolidate styles dynamically to ensure both modes share EXACTLY the same layout
    if is_dark:
        bg_app = "#0e1117"
        bg_app_image = "radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%)"
        bg_sidebar = "rgba(10, 11, 26, 0.65)"
        bg_card = "linear-gradient(145deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%)"
        bg_input = "rgba(15, 23, 42, 0.6)"
        
        text_primary = "#e2e8f0"
        text_muted = "#94a3b8"
        
        border_color = "rgba(255, 255, 255, 0.08)"
        border_input = "rgba(255, 255, 255, 0.1)"
        
        primary_accent = "#8b5cf6"
        btn_bg = "linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%)"
        btn_shadow = "rgba(99, 102, 241, 0.3)"
        
        h1_gradient = "linear-gradient(90deg, #FFFFFF 0%, #a5b4fc 100%)"
        metric_gradient = "linear-gradient(90deg, #2ec4b6 0%, #2bd9fe 100%)"
        
        nav_active_bg = "rgba(139, 92, 246, 0.1)"
        nav_active_text = "#a5b4fc"
        nav_hover = "rgba(255, 255, 255, 0.05)"
    else:
        bg_app = "#f8fafc"
        bg_app_image = "radial-gradient(at 0% 0%, hsla(210,100%,98%,1) 0, transparent 50%), radial-gradient(at 50% 0%, hsla(240,80%,98%,1) 0, transparent 50%), radial-gradient(at 100% 0%, hsla(280,70%,96%,1) 0, transparent 50%)"
        bg_sidebar = "rgba(255, 255, 255, 0.95)"
        bg_card = "linear-gradient(145deg, rgba(255, 255, 255, 1) 0%, rgba(248, 250, 252, 1) 100%)"
        bg_input = "#ffffff"
        
        text_primary = "#0f172a"
        text_muted = "#64748b"
        
        border_color = "rgba(203, 213, 225, 1)"
        border_input = "rgba(148, 163, 184, 1)"
        
        primary_accent = "#4f46e5"
        btn_bg = "linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%)"
        btn_shadow = "rgba(79, 70, 229, 0.2)"
        
        h1_gradient = "linear-gradient(90deg, #1e293b 0%, #4338ca 100%)"
        metric_gradient = "linear-gradient(90deg, #0d9488 0%, #0284c7 100%)"
        
        nav_active_bg = "rgba(79, 70, 229, 0.1)"
        nav_active_text = "#4f46e5"
        nav_hover = "#f1f5f9"

    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
        
        html, body, [class*="css"] {{ font-family: 'Outfit', 'Inter', sans-serif; }}
        
        .stApp {{
            background-color: {bg_app};
            background-image: {bg_app_image};
            background-attachment: fixed;
            background-size: cover;
            color: {text_primary} !important;
        }}
        
        /* Targeted safe overrides avoiding Streamlit widget toggle icons */
        h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown li, [data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p, label p {{
            color: {text_primary} !important;
        }}
        
        [data-testid="stSidebar"] {{
            background-color: {bg_sidebar} !important; 
            backdrop-filter: blur(24px); 
            border-right: 1px solid {border_color} !important;
        }}
        
        .stCard, div[data-testid="stExpander"], div.stContainer {{
            background: {bg_card} !important;
            border: 1px solid {border_color} !important; 
            border-radius: 20px; 
            padding: 24px;
            box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.05); 
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }}
        .stCard:hover, div.stContainer:hover {{ border-color: {primary_accent} !important; box-shadow: 0 12px 40px -8px {nav_active_bg}; transform: translateY(-4px); }}
        
        h1, h2, h3, h4, h5, h6 {{ font-weight: 700 !important; letter-spacing: -0.02em; }}
        h1 {{ background: {h1_gradient}; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; font-size: 3rem !important; padding-bottom: 10px; }}
        
        .stButton > button {{
            background: {btn_bg} !important; color: white !important; border: none !important; border-radius: 12px;
            padding: 0.75rem 2rem; font-weight: 600; box-shadow: 0 4px 15px {btn_shadow}; transition: all 0.3s ease;
        }}
        .stButton > button:hover {{ transform: translateY(-2px); }}
        
        .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div, section[data-testid="stFileUploadDropzone"] {{
            background-color: {bg_input} !important; border: 1px solid {border_input} !important; border-radius: 12px; transition: border 0.2s;
        }}
        section[data-testid="stFileUploadDropzone"] * {{ color: {text_primary} !important; }}
        
        input, textarea, select, .stSelectbox [data-baseweb="select"] * {{ color: {text_primary} !important; background: transparent !important; }}
        ::placeholder, ::-webkit-input-placeholder, :-ms-input-placeholder {{ color: {text_muted} !important; opacity: 1; }}
        
        .stTextInput > div > div:focus-within, .stTextArea > div > div:focus-within, section[data-testid="stFileUploadDropzone"]:focus-within {{
            border-color: {primary_accent} !important; box-shadow: 0 0 0 2px {nav_active_bg} !important;
        }}
        
        [data-testid="stMetricValue"] {{
            font-family: 'Outfit', sans-serif; font-size: 2.8rem !important; font-weight: 700 !important;
            background: {metric_gradient}; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
        }}
        
        div[role="radiogroup"] > label {{
            padding: 10px 15px; border-radius: 10px; transition: background 0.3s;
            color: {text_primary} !important; font-weight: 600 !important;
        }}
        div[role="radiogroup"] p {{ color: {text_primary} !important; }}
        div[role="radiogroup"] > label:hover {{ background: {nav_hover} !important; }}
        div[role="radiogroup"] > label[data-checked="true"] {{ background: {nav_active_bg} !important; color: {nav_active_text} !important; }}
        div[role="radiogroup"] > label[data-checked="true"] p, div[role="radiogroup"] > label[data-checked="true"] div {{ color: {nav_active_text} !important; }}
        
        hr {{ border-color: {border_color} !important; margin: 2em 0; }}
        .stAlert {{ background-color: {bg_card} !important; border: 1px solid {border_color} !important; border-radius: 16px; color: {text_primary} !important; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────────────
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "candidate_history" not in st.session_state:
    st.session_state.candidate_history = []


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='font-size: 1.5rem; margin-bottom: 0px;'>SmartHire AI</h1>", unsafe_allow_html=True)
    st.caption("Next-Gen Hiring Intelligence")
    
    # Theme Toggle
    is_dark_mode = st.toggle("🌙 Dark Mode", value=True)
    inject_custom_css(is_dark=is_dark_mode)
    
    st.markdown("---")
    
    # Navigation Radio Instead of Tabs
    nav_options = [
        "Input", "Analysis", "Prediction", "Explain",
        "Growth", "Interview", "Fairness", "Rank",
        "Insights", "Report"
    ]
    active_tab = st.radio("Navigation", nav_options, label_visibility="collapsed")
    
    st.markdown("---")
    
    st.markdown("**Core Configuration**")
    model_choice = st.selectbox("Predictive Model", 
                                ["Ensemble (Recommended)", "XGBoost", "Random Forest", "SVM", "Logistic Regression"])
    
    # Fake session stats for visual
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Credits", "∞")
    c2.metric("Uptime", "99.9%")
    
    st.markdown("---")
    with st.expander("ℹ️  System Info"):
        st.markdown("""
        **Version 2.0**
        • Engine: `Scikit-Learn`
        • Explainer: `SHAP`
        • UI: `Streamlit Custom`
        """)


# ── TAB 1: Home (Input) ──────────────────────────────────────────────────────
if active_tab == "Input":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; margin-bottom: 12px;'>Building the Future, One Hire at a Time.</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 48px; font-size: 1.2rem; color: #94a3b8;'>AI-powered recruitment analysis with deep learning precision.</p>", unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### 01. Resume Upload")
            uploaded_file = st.file_uploader("Drop PDF or DOCX", type=["pdf", "docx", "txt"])
            
            resume_text = ""
            if uploaded_file:
                resume_text = parse_resume(uploaded_file, uploaded_file.name)
                if len(resume_text) > 50:
                    st.toast("Resume uploaded successfully", icon="⚡")
                    with st.expander("📄 View Parsed Content"):
                        st.caption(resume_text[:500] + "...")
                else:
                    st.error("File is empty or unreadable.")
        
        with col2:
            st.markdown("### 02. Role Requirements")
            jd_text = st.text_area("Job Description (JD)", height=300, 
                                   placeholder="Paste the detailed job description here...")
            
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if st.button("Initialize Analysis Sequence", type="primary", use_container_width=True):
            if not resume_text or not jd_text:
                st.warning("⚠️ Please provide both a resume and a job description.")
            else:
                with st.spinner("Processing NLP Layer & Computing Embeddings..."):
                    # 1. Processing
                    entities = extract_all_entities(resume_text)
                    skill_gap = compute_skill_gap(resume_text, jd_text)
                    similarity = combined_similarity(resume_text, jd_text)
                    scores = compute_weighted_score(resume_text, jd_text)
                    bias_result = detect_bias_flags(resume_text)
                    
                    # 2. ML Prediction
                    feat_dict = {
                        "skill_match_score": skill_gap["match_score"] * 100,
                        "experience_years": entities["experience_years"],
                        "education_score": scores["components"]["education"]["raw"] * 100,
                        "project_score": scores["components"]["projects"]["raw"] * 100,
                        "bert_similarity": similarity["bert_score"],
                        "tfidf_similarity": similarity["tfidf_score"],
                        "resume_length_score": min(len(resume_text.split())/1000, 1.0),
                        "skills_count": len(entities["skills"]),
                    }
                    
                    fv = get_feature_vector(feat_dict)
                    ml_results = predict_all_models(fv)
                    
                    # 3. Explainer
                    shap_res = compute_shap_values(fv)
                    
                    # 4. Feature modules
                    recs = generate_learning_path(skill_gap["missing_skills"])
                    questions = generate_questions(skill_gap["matched_skills"], resume_text)
                    
                    # Store in session state
                    candidate_name = entities["persons"][0] if entities["persons"] else "Candidate"
                    
                    # Determine selected model result
                    model_key = model_choice.lower().replace(" ", "_").replace("(recommended)", "").strip()
                    if "ensemble" in model_key: model_key = "ensemble"
                    
                    final_pred = ml_results[model_key]
                    
                    full_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "candidate_name": candidate_name,
                        "job_title": extract_noun_phrases(jd_text)[0].title() if extract_noun_phrases(jd_text) else "Role",
                        "resume_text": resume_text,
                        "jd_text": jd_text,
                        "entities": entities,
                        "skill_gap": skill_gap,
                        "similarity": similarity,
                        "scores": scores,
                        "bias_result": bias_result,
                        "features": feat_dict,
                        "ml_results": ml_results,
                        "shap_data": shap_res,
                        "recommendations": recs,
                        "interview_questions": questions,
                        
                        # Flat fields for ranking/reports
                        "ai_score": scores["final_score"],
                        "selection_prob": final_pred["probability_selected"],
                        "prediction": final_pred["prediction"],
                        "skill_match_percent": skill_gap["match_percent"],
                        "matched_skills": skill_gap["matched_skills"],
                        "missing_skills": skill_gap["missing_skills"],
                        "experience_years": entities["experience_years"],
                        "education": ", ".join(entities["education"][:2]),
                        "bert_score": similarity["bert_score"],
                    }
                    
                    st.session_state.analysis_data = full_data
                    
                    # Add to history if unique
                    if not any(c["resume_text"] == resume_text for c in st.session_state.candidate_history):
                        st.session_state.candidate_history.append(full_data)
                    
                    st.toast("Analysis Sequence Complete", icon="🚀")
                    time.sleep(1)
                    st.rerun()


# ── DATA CHECK ──────────────────────────────────────────────────────────────
data = st.session_state.analysis_data

if not data and active_tab != "Input":
    st.info("👈 Please initialize analysis from the 'Input' tab.")
else:
    # ── TAB 2: Analysis ──────────────────────────────────────────────────────
    if active_tab == "Analysis":
        st.header(f"Analysis: {data['candidate_name']}")
        
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AI Score", f"{data['ai_score']}/100", 
                  delta=f"{data['ai_score']-50:.1f} vs Avg" if data['ai_score']>50 else None)
        m2.metric("Skill Match", f"{data['skill_match_percent']}%")
        m3.metric("Experience", f"{data['entities']['experience_years']} Years")
        m4.metric("Semantic Match", f"{data['similarity']['combined_percent']}%")
        
        st.markdown("---")
        
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Skill Gap Breakdown")
            
            s1, s2 = st.columns(2)
            with s1:
                st.success(f"✅ Matched ({len(data['skill_gap']['matched_skills'])})")
                st.write(", ".join(data['skill_gap']['matched_skills']) or "None")
            with s2:
                st.error(f"❌ Missing ({len(data['skill_gap']['missing_skills'])})")
                st.write(", ".join(data['skill_gap']['missing_skills']) or "None")
                
        with c2:
            st.subheader("Component Scoring")
            scores = data['scores']['components']
            df_scores = pd.DataFrame({
                "Component": ["Skill Match", "Experience", "Education", "Projects"],
                "Score": [scores["skill_match"]["raw"]*100, scores["experience"]["raw"]*100, 
                          scores["education"]["raw"]*100, scores["projects"]["raw"]*100]
            })
            fig = px.bar(df_scores, x="Score", y="Component", orientation='h', 
                         color="Score", color_continuous_scale="mrybm", range_x=[0, 100])
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), 
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={'color': "#94a3b8"})
            st.plotly_chart(fig, use_container_width=True)


    # ── TAB 3: Prediction ────────────────────────────────────────────────────
    if active_tab == "Prediction":
        st.header("Prediction Engine")
        
        sel_prob = data['selection_prob']
        pred_label = data['prediction']
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### AI Decision")
            if pred_label == "Selected":
                st.success(f"### ✅ {pred_label}")
            else:
                st.error(f"### ❌ {pred_label}")
                
            st.markdown(f"**Confidence:** {round(sel_prob * 100, 1)}%")
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sel_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Selection Probability"},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#8b5cf6" if sel_prob > 0.5 else "#f43f5e"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(244, 63, 94, 0.2)"},
                        {'range': [50, 100], 'color': "rgba(139, 92, 246, 0.2)"}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), 
                                    paper_bgcolor="rgba(0,0,0,0)", font={'color': "#e2e8f0"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.markdown("### Ensemble Consensus")
            res = data['ml_results']
            models = ["logistic_regression", "svm", "random_forest", "xgboost", "ensemble"]
            
            rows = []
            for m in models:
                prob = res[m]['probability_selected']
                rows.append({
                    "Model": m.replace("_", " ").title(), 
                    "Probability": prob,
                    "Decision": "✅ Select" if prob > 0.5 else "❌ Reject"
                })
            
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


    # ── TAB 4: Explainability ────────────────────────────────────────────────
    if active_tab == "Explain":
        st.header("Decision Intelligence (SHAP)")
        st.markdown("Feature contribution analysis for transparent decision-making.")
        
        if "error" in data['shap_data']:
            st.error("Could not generate SHAP values.")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Feature Impact Cloud")
                import io
                from PIL import Image
                buf = io.BytesIO(plot_shap_bar(data['shap_data']))
                if buf.getbuffer().nbytes > 0:
                    st.image(Image.open(buf), use_column_width=True)
                else:
                    st.warning("Not enough data to plot SHAP values.")
            
            with col2:
                st.subheader("Key Drivers")
                factors = get_top_factors(data['shap_data'])
                for f in factors:
                    color = "#4ade80" if f['direction'] == "positive" else "#f87171"
                    icon = "⬆️" if f['direction'] == "positive" else "⬇️"
                    st.markdown(f"**{icon} {f['feature']}**")
                    st.markdown(f"<span style='color:{color}'>Impact: {f['shap_value']}</span>", unsafe_allow_html=True)
                    st.divider()


    # ── TAB 5: Recommendations ───────────────────────────────────────────────
    if active_tab == "Growth":
        st.header("Growth Plan")
        
        recs = data['recommendations']
        if not recs['total_skills_recommended']:
            st.success("🎉 Outstanding profile! No critical skill gaps identified.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("🚀 Immediate Action")
                for r in recs['phase_1_immediate']:
                    with st.container(border=True):
                        st.markdown(f"**{r['skill'].title()}**")
                        st.caption(r['description'])
                        st.link_button("View Resource", r['resource_url'])
            
            with c2:
                st.subheader("📅 Short Term")
                for r in recs['phase_2_short_term']:
                    with st.container(border=True):
                        st.markdown(f"**{r['skill'].title()}**")
                        st.caption(r['description'])
            
            with c3:
                st.subheader("🔭 Long Term")
                for r in recs['phase_3_long_term']:
                    with st.container(border=True):
                        st.markdown(f"**{r['skill'].title()}**")
                        st.caption(r['description'])


    # ── TAB 6: Interview ─────────────────────────────────────────────────────
    if active_tab == "Interview":
        st.header("Generated Interview Questions")
        
        questions = data['interview_questions']
        
        for i, q in enumerate(questions):
            with st.expander(f"Q{i+1}: {q['question']}  [{q['type']}]"):
                st.caption(f"Category: {q['category']}")
                st.text_area("Candidate Answer / Notes:", key=f"ans_{i}")
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.number_input("Rating (1-10)", min_value=1, max_value=10, key=f"rate_{i}")


    # ── TAB 7: Bias Check ────────────────────────────────────────────────────
    if active_tab == "Fairness":
        st.header("Fairness & Bias Audit")
        
        bias = data['bias_result']
        fair_score = bias['fairness_confidence_score']
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Fairness Score", f"{fair_score}/100")
            if fair_score > 80:
                st.success("Low Bias Risk Detected")
            elif fair_score > 50:
                st.warning("Medium Bias Risk")
            else:
                st.error("High Bias Risk Detected")
                
        with c2:
            st.subheader("Risk Indicators")
            if not bias['flags']:
                st.info("No explicit bias indicators found in the resume text.")
            else:
                for key, val in bias['flags'].items():
                    with st.expander(f"⚠️ {key.replace('_', ' ').title()}"):
                        st.write(f"**Found:** {', '.join(val['found'])}")
                        st.write(f"**Risk Level:** {val['risk']}")
                        st.write(f"**Note:** {val['note']}")


    # ── TAB 8: Leaderboard ───────────────────────────────────────────────────
    if active_tab == "Rank":
        st.header("Candidate Leaderboard")
        
        history = st.session_state.candidate_history
        if len(history) < 2:
            st.info("Process more candidates to populate the leaderboard.")
        else:
            ranked = rank_candidates(history)
            
            rows = []
            for c in ranked:
                rows.append({
                    "Rank": c['rank'],
                    "Name": c['candidate_name'],
                    "Tier": c['tier'],
                    "AI Score": c['ai_score'],
                    "Sel. Prob %": round(c['selection_prob']*100, 1),
                    "Skill Match %": c['skill_match_percent'],
                    "Composite Score": c['composite_score']
                })
            
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


    # ── TAB 9: Analytics ─────────────────────────────────────────────────────
    if active_tab == "Insights":
        st.header("HR Analytics Dashboard")
        
        metrics = get_model_metrics()
        if not metrics:
            st.warning("Model metrics not found. Please train models first.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Model Accuracy")
                df_m = pd.DataFrame(metrics).T.reset_index()
                df_m = pd.melt(df_m, id_vars="index", var_name="Metric", value_name="Score")
                fig = px.bar(df_m, x="index", y="Score", color="Metric", barmode="group",
                             title="Model Performance Metrics")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font={'color': "#94a3b8"})
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.subheader("Pipeline Funnel")
                funnel_data = dict(
                    number=[1200, 800, 400, 150, 50],
                    stage=["Applied", "Keyword Match", "AI Screened", "Interviewed", "Hired"]
                )
                fig_f = px.funnel(funnel_data, x='number', y='stage')
                fig_f.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#94a3b8"})
                st.plotly_chart(fig_f, use_container_width=True)


    # ── TAB 10: Report ───────────────────────────────────────────────────────
    if active_tab == "Report":
        st.header("Export Report")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"Generate a comprehensive PDF dossier for **{data['candidate_name']}**.")
            st.markdown("""
            **Dossier Contents:**
            - Executive Summary
            - Full Skill Gap Analysis
            - ML Prediction Confidence
            - Interview Questions
            """)
        
        with c2:
            st.write("")
            st.write("")
            pdf_bytes = generate_pdf_report(data)
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download PDF Dossier",
                    data=pdf_bytes,
                    file_name=f"SmartHire_Report_{data['candidate_name'].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            else:
                st.error("Error generating PDF.")
