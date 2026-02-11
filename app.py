import streamlit as st

from src.analyzer import analyze_resume_against_jd
from src.config import get_settings
from src.suggester import build_suggestions
from src.text_extract import extract_resume_text

st.set_page_config(page_title="Resume JD Matcher", page_icon="page_with_curl", layout="wide")

st.title("Resume JD Matcher")
st.caption("Upload a resume, paste a job description, and get a concrete match analysis.")
settings = get_settings()

with st.form("matcher_form"):
    col1, col2 = st.columns([1, 2])
    with col1:
        resume_file = st.file_uploader("Resume file", type=["pdf", "docx", "txt"])
        use_hybrid_llm = st.checkbox(
            "Use hybrid LLM matching",
            value=True,
            help="Combine deterministic scoring with LLM semantic assessment.",
        )
        use_llm_suggestions = st.checkbox(
            "Use OpenRouter for rewrite suggestions",
            value=False,
            help="Requires OPENROUTER_API_KEY in .env",
        )

    with col2:
        jd_text = st.text_area(
            "Job Description",
            height=260,
            placeholder="Paste the full JD here...",
        )

    analyze_clicked = st.form_submit_button("Analyze match", type="primary")

if analyze_clicked:
    if resume_file is None:
        st.error("Upload a resume file first.")
        st.stop()
    if not jd_text.strip():
        st.error("Paste a job description first.")
        st.stop()

    with st.spinner("Parsing resume and analyzing JD match..."):
        try:
            resume_text = extract_resume_text(resume_file)
        except Exception as exc:
            st.error(f"Failed to parse resume file: {exc}")
            st.stop()

        if len(resume_text) < 60:
            st.error("Could not extract enough text from resume. Try a cleaner PDF/DOCX.")
            st.stop()

        analysis = analyze_resume_against_jd(
            resume_text=resume_text,
            jd_text=jd_text,
            use_llm_matching=use_hybrid_llm and bool(settings.openrouter_api_key),
        )
        suggestions = build_suggestions(
            resume_text=resume_text,
            jd_text=jd_text,
            analysis=analysis,
            use_llm=use_llm_suggestions,
        )

    st.subheader("Overall Match")
    score = analysis["overall_score"]
    st.metric("Match Score", f"{score}/100")
    st.progress(min(max(score, 0), 100) / 100)
    st.caption(f"Scoring mode: **{analysis.get('scoring_mode', 'deterministic')}**")
    if analysis.get("scoring_mode") == "hybrid":
        st.caption(f"Deterministic baseline: **{analysis.get('deterministic_score', score)}/100**")
    if analysis.get("llm_warning"):
        st.warning(analysis["llm_warning"])

    has_llm_component = "llm_alignment" in analysis["components"]
    if has_llm_component:
        c1, c2, c3, c4 = st.columns(4)
    else:
        c1, c2, c3 = st.columns(3)
        c4 = None

    c1.metric("Keyword Overlap", f"{analysis['components']['keyword_overlap']}%")
    c2.metric("Semantic Similarity", f"{analysis['components']['semantic_similarity']}%")
    c3.metric("Required Skill Coverage", f"{analysis['components']['required_skill_coverage']}%")
    if has_llm_component and c4:
        c4.metric("LLM Alignment", f"{analysis['components']['llm_alignment']}%")

    llm_eval = analysis.get("llm_evaluation")
    if llm_eval:
        with st.expander("LLM Matching Insights"):
            st.markdown(f"**Semantic alignment:** {llm_eval['semantic_alignment']}%")
            st.markdown(f"**Skills alignment:** {llm_eval['skills_alignment']}%")
            st.markdown(f"**Evidence quality:** {llm_eval['evidence_quality']}%")
            if llm_eval.get("missing_areas"):
                st.markdown("**LLM-detected gaps:** " + ", ".join(llm_eval["missing_areas"]))
            if llm_eval.get("reasoning"):
                st.markdown("**Reasoning:**")
                st.write(llm_eval["reasoning"])

    st.subheader("Skills Coverage")
    req = analysis["required_skills"]
    matched = analysis["matched_skills"]
    missing = analysis["missing_skills"]

    left, right = st.columns(2)
    with left:
        st.markdown("**Required skills detected from JD**")
        st.write(", ".join(req) if req else "No explicit skills detected from JD text.")
        st.markdown("**Matched skills in resume**")
        st.write(", ".join(matched) if matched else "None detected.")
    with right:
        st.markdown("**Missing skills**")
        st.write(", ".join(missing) if missing else "No major skill gaps detected.")

    st.subheader("Missing Keywords")
    missing_keywords = analysis["missing_keywords"]
    if missing_keywords:
        st.write(", ".join(missing_keywords))
    else:
        st.write("No major keyword gaps detected.")

    st.subheader("Section-wise Feedback")
    for section, item in analysis["section_feedback"].items():
        st.markdown(f"**{section}: {item['level']}**")
        st.write(item["comment"])

    st.subheader("Rewrite Suggestions")
    for idx, suggestion in enumerate(suggestions, start=1):
        st.markdown(f"{idx}. {suggestion}")

    with st.expander("Extracted Resume Text (preview)"):
        st.text(resume_text[:3000])
