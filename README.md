# Resume JD Matcher (Hybrid AI + Rule-Based Scoring)

Resume JD Matcher is a Streamlit application that evaluates how well a resume aligns with a job description.  
It combines deterministic scoring (keywords, similarity, skill coverage) with optional LLM-based semantic evaluation for a hybrid match score.

## Core Features

- Resume parsing for `PDF`, `DOCX`, and `TXT`
- Job description analysis and skill extraction
- Deterministic scoring with component breakdown:
  - keyword overlap
  - semantic similarity
  - required skill coverage
- Optional hybrid LLM matching:
  - semantic alignment
  - skills alignment
  - evidence quality
- Section-wise feedback (`Skills`, `Projects`, `Experience`)
- Missing keywords and missing required skills
- Rewrite suggestions (rule-based + optional LLM suggestions)

## Scoring Logic

Deterministic score:

- `40%` keyword overlap
- `35%` semantic similarity
- `25%` required skill coverage

Hybrid mode (optional):

- LLM computes an additional alignment score
- Final score = `65% deterministic + 35% LLM alignment`

## Tech Stack

- Python 3.14
- Streamlit
- pypdf
- python-docx
- OpenRouter (OpenAI-compatible API via `openai` SDK)

## Project Structure

```text
.
+-- app.py
+-- requirements.txt
+-- .env.example
+-- src
|   +-- __init__.py
|   +-- analyzer.py
|   +-- config.py
|   +-- suggester.py
|   +-- text_extract.py
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Update `.env` if LLM features are required:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
API_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
```

## Run

```powershell
streamlit run app.py --server.port 8503
```

## Usage

1. Upload a resume (`.pdf`, `.docx`, or `.txt`).
2. Paste the target job description.
3. Enable `Use hybrid LLM matching` for hybrid scoring (optional).
4. Click `Analyze match`.
5. Review score components, gaps, feedback, and suggestions.
