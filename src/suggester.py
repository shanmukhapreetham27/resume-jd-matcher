from __future__ import annotations

from typing import Any

from openai import OpenAI

from .config import get_settings


def rule_based_suggestions(analysis: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []

    missing_skills = analysis.get("missing_skills", [])
    missing_keywords = analysis.get("missing_keywords", [])
    components = analysis.get("components", {})

    if missing_skills:
        top_skills = ", ".join(missing_skills[:5])
        suggestions.append(
            f"Add explicit mentions of missing required skills in Skills/Projects: {top_skills}."
        )

    if components.get("keyword_overlap", 0) < 55 and missing_keywords:
        kw = ", ".join(missing_keywords[:6])
        suggestions.append(
            f"Include more JD-specific terms in resume bullets (naturally, not as a keyword dump): {kw}."
        )

    if components.get("semantic_similarity", 0) < 60:
        suggestions.append(
            "Rewrite project bullets to mirror JD language and responsibilities (tools, use-cases, outcomes)."
        )

    section_feedback = analysis.get("section_feedback", {})
    if section_feedback.get("Projects", {}).get("level") != "Strong":
        suggestions.append(
            "Use action-first bullets with measurable impact, e.g., 'Built X using Y, improved Z by N%'."
        )

    if section_feedback.get("Experience", {}).get("level") == "Needs improvement":
        suggestions.append(
            "Add one internship/research/leadership subsection with 2-3 quantified bullets."
        )

    if not suggestions:
        suggestions.append("Good alignment overall. Fine-tune by tailoring 2-3 bullets directly to the JD.")

    return suggestions[:5]


def llm_rewrite_suggestions(resume_text: str, jd_text: str, analysis: dict[str, Any]) -> list[str]:
    settings = get_settings()
    if not settings.openrouter_api_key:
        return []

    client = OpenAI(api_key=settings.openrouter_api_key, base_url=settings.api_base_url)
    prompt = f"""
You are a resume reviewer.
Given a resume and job description, provide exactly 3 concise rewrite suggestions.

Constraints:
- Internship-level resume
- Action-oriented bullet style
- Include measurable outcomes where possible
- Do not invent fake achievements

Job Description:
{jd_text[:5000]}

Resume:
{resume_text[:7000]}

Current analysis snapshot:
{analysis}

Return only 3 bullet points.
""".strip()

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
    except Exception:
        return []

    cleaned = [line.strip("-• \n\t") for line in content.splitlines() if line.strip()]
    # Keep meaningful lines only.
    cleaned = [line for line in cleaned if len(line) > 12]
    return cleaned[:3]


def build_suggestions(resume_text: str, jd_text: str, analysis: dict[str, Any], use_llm: bool) -> list[str]:
    base = rule_based_suggestions(analysis)
    if not use_llm:
        return base

    llm_items = llm_rewrite_suggestions(resume_text, jd_text, analysis)
    return llm_items if llm_items else base
