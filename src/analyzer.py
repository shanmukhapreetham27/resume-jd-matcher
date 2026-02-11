from __future__ import annotations

from collections import Counter
import json
import math
import re
from typing import Any

from openai import OpenAI

from .config import get_settings

# Intern-friendly skill dictionary to estimate required-skill coverage.
SKILL_CATALOG = [
    "python",
    "sql",
    "excel",
    "power bi",
    "tableau",
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "llm",
    "rag",
    "prompt engineering",
    "data analysis",
    "data visualization",
    "statistics",
    "regression",
    "classification",
    "time series",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "aws",
    "azure",
    "gcp",
    "docker",
    "git",
    "github",
    "api",
    "fastapi",
    "streamlit",
    "communication",
    "stakeholder",
    "experimentation",
    "a/b testing",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "will",
    "you",
    "your",
    "we",
    "our",
    "this",
    "these",
    "those",
    "using",
    "use",
    "used",
    "build",
    "built",
    "ability",
    "work",
    "working",
    "candidate",
    "role",
    "team",
}

ACTION_VERBS = {
    "built",
    "developed",
    "designed",
    "implemented",
    "analyzed",
    "optimised",
    "optimized",
    "deployed",
    "improved",
    "led",
    "created",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.\-]{2,}", normalize_text(text))


def tokenize_with_bigrams(text: str) -> list[str]:
    tokens = tokenize(text)
    bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def extract_keywords(text: str, top_n: int = 35) -> list[str]:
    tokens = [
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    ]
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(top_n)]


def extract_required_skills(jd_text: str) -> list[str]:
    normalized = normalize_text(jd_text)
    found = []
    for skill in SKILL_CATALOG:
        skill_pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(skill_pattern, normalized):
            found.append(skill)
    return sorted(set(found))


def semantic_similarity_score(resume_text: str, jd_text: str) -> float:
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    docs = [tokenize_with_bigrams(resume_text), tokenize_with_bigrams(jd_text)]
    term_counts = [Counter(doc) for doc in docs]

    vocabulary = set(term_counts[0].keys()) | set(term_counts[1].keys())
    if not vocabulary:
        return 0.0

    # Document frequency and smooth IDF for two-doc corpus.
    df = {}
    for term in vocabulary:
        df[term] = int(term in term_counts[0]) + int(term in term_counts[1])

    n_docs = 2
    idf = {term: math.log((1 + n_docs) / (1 + df_val)) + 1.0 for term, df_val in df.items()}

    def build_tfidf_vector(counter: Counter[str]) -> dict[str, float]:
        total = sum(counter.values()) or 1
        return {term: (count / total) * idf[term] for term, count in counter.items()}

    vec_a = build_tfidf_vector(term_counts[0])
    vec_b = build_tfidf_vector(term_counts[1])

    dot = 0.0
    for term, val in vec_a.items():
        dot += val * vec_b.get(term, 0.0)

    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = dot / (norm_a * norm_b)
    return float(max(0.0, min(1.0, sim)))


def keyword_overlap_score(resume_text: str, jd_keywords: list[str]) -> tuple[float, list[str]]:
    resume_tokens = set(tokenize(resume_text))
    jd_set = set(jd_keywords)
    if not jd_set:
        return 0.0, []

    matched = sorted([kw for kw in jd_set if kw in resume_tokens])
    score = len(matched) / len(jd_set)
    return score, matched


def skills_coverage_score(resume_text: str, required_skills: list[str]) -> tuple[float, list[str]]:
    if not required_skills:
        return 1.0, []

    normalized_resume = normalize_text(resume_text)
    matched = []
    for skill in required_skills:
        if re.search(r"\b" + re.escape(skill) + r"\b", normalized_resume):
            matched.append(skill)

    score = len(matched) / len(required_skills)
    return score, sorted(matched)


def section_feedback(resume_text: str, required_skills: list[str], matched_skills: list[str]) -> dict[str, dict[str, str]]:
    lower_resume = normalize_text(resume_text)
    numbers_count = len(re.findall(r"\b\d+(?:\.\d+)?%?\b|\b\d+[km]\+?\b", lower_resume))
    verbs_count = sum(1 for verb in ACTION_VERBS if re.search(r"\b" + re.escape(verb) + r"\b", lower_resume))

    skill_ratio = (len(matched_skills) / len(required_skills)) if required_skills else 1.0
    if skill_ratio >= 0.8:
        skills_msg = "Strong skill alignment with job requirements."
        skills_level = "Strong"
    elif skill_ratio >= 0.5:
        skills_msg = "Moderate skill match; add missing tools/skills explicitly."
        skills_level = "Moderate"
    else:
        skills_msg = "Low required-skill coverage; prioritize core JD skills in Skills/Projects."
        skills_level = "Needs improvement"

    if numbers_count >= 5 and verbs_count >= 5:
        projects_msg = "Project bullets are impact-oriented and quantified."
        projects_level = "Strong"
    elif numbers_count >= 2 and verbs_count >= 3:
        projects_msg = "Projects are decent; add more measurable outcomes."
        projects_level = "Moderate"
    else:
        projects_msg = "Projects need stronger action verbs and quantified impact."
        projects_level = "Needs improvement"

    has_experience_signals = bool(
        re.search(r"\b(intern|experience|worked|responsib|led|managed|collaborat)\w*\b", lower_resume)
    )
    if has_experience_signals and numbers_count >= 3:
        exp_msg = "Experience signals are present and backed by outcomes."
        exp_level = "Strong"
    elif has_experience_signals:
        exp_msg = "Experience exists but impact quantification can improve."
        exp_level = "Moderate"
    else:
        exp_msg = "Add internship/leadership experience style bullets to improve credibility."
        exp_level = "Needs improvement"

    return {
        "Skills": {"level": skills_level, "comment": skills_msg},
        "Projects": {"level": projects_level, "comment": projects_msg},
        "Experience": {"level": exp_level, "comment": exp_msg},
    }


def _safe_int(value: Any, default: int = 50) -> int:
    try:
        return int(max(0, min(100, int(value))))
    except Exception:
        return default


def _parse_json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def llm_alignment_assessment(resume_text: str, jd_text: str) -> dict[str, Any] | None:
    settings = get_settings()
    if not settings.openrouter_api_key:
        return None

    client = OpenAI(api_key=settings.openrouter_api_key, base_url=settings.api_base_url)
    prompt = f"""
You are evaluating a resume for a job description.
Return ONLY JSON in this exact format:
{{
  "semantic_alignment": 0,
  "skills_alignment": 0,
  "evidence_quality": 0,
  "missing_areas": ["", ""],
  "reasoning": ""
}}

Scoring rules:
- semantic_alignment: how well resume content matches role responsibilities (0-100)
- skills_alignment: required tools/skills coverage (0-100)
- evidence_quality: quantified impact, clarity, ownership in bullets (0-100)

Keep missing_areas to max 5 concise items.
Do not include markdown, extra text, or code fences.

JOB DESCRIPTION:
{jd_text[:7000]}

RESUME:
{resume_text[:9000]}
""".strip()

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        parsed = _parse_json_from_text(content)
        if not parsed:
            return None

        semantic = _safe_int(parsed.get("semantic_alignment"), default=50)
        skills = _safe_int(parsed.get("skills_alignment"), default=50)
        evidence = _safe_int(parsed.get("evidence_quality"), default=50)

        missing = parsed.get("missing_areas", [])
        if not isinstance(missing, list):
            missing = []
        missing = [str(item).strip() for item in missing if str(item).strip()][:5]

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_overall = int(round((0.50 * semantic) + (0.30 * skills) + (0.20 * evidence)))

        return {
            "semantic_alignment": semantic,
            "skills_alignment": skills,
            "evidence_quality": evidence,
            "llm_overall": llm_overall,
            "missing_areas": missing,
            "reasoning": reasoning,
        }
    except Exception:
        return None


def analyze_resume_against_jd(resume_text: str, jd_text: str, use_llm_matching: bool = False) -> dict[str, Any]:
    jd_keywords = extract_keywords(jd_text, top_n=40)
    required_skills = extract_required_skills(jd_text)

    keyword_score, matched_keywords = keyword_overlap_score(resume_text, jd_keywords)
    semantic_score = semantic_similarity_score(resume_text, jd_text)
    skills_score, matched_skills = skills_coverage_score(resume_text, required_skills)

    missing_keywords = [kw for kw in jd_keywords if kw not in matched_keywords]
    missing_skills = [skill for skill in required_skills if skill not in matched_skills]

    # Deterministic baseline (stable and reproducible).
    weighted_score = (0.40 * keyword_score) + (0.35 * semantic_score) + (0.25 * skills_score)
    deterministic_score = int(round(weighted_score * 100))
    overall_score = deterministic_score

    components = {
        "keyword_overlap": int(round(keyword_score * 100)),
        "semantic_similarity": int(round(semantic_score * 100)),
        "required_skill_coverage": int(round(skills_score * 100)),
    }

    result = {
        "overall_score": overall_score,
        "deterministic_score": deterministic_score,
        "scoring_mode": "deterministic",
        "components": components,
        "jd_keywords": jd_keywords,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords[:20],
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "section_feedback": section_feedback(resume_text, required_skills, matched_skills),
    }

    if use_llm_matching:
        llm_eval = llm_alignment_assessment(resume_text, jd_text)
        if llm_eval:
            hybrid_score = int(round((0.65 * deterministic_score) + (0.35 * llm_eval["llm_overall"])))
            result["overall_score"] = hybrid_score
            result["scoring_mode"] = "hybrid"
            result["llm_evaluation"] = llm_eval
            result["components"]["llm_alignment"] = llm_eval["llm_overall"]
        else:
            result["llm_evaluation"] = None
            result["llm_warning"] = "LLM matching unavailable; showing deterministic score."

    return result
