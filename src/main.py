import json
import os
import sys

import time
import random
from google.genai.errors import ServerError

from pathlib import Path
from dotenv import load_dotenv

import pdfplumber

from google import genai
from google.genai import types

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESUME_PATH = BASE_DIR / "resume.pdf"
OUTPUT_DIR = BASE_DIR / "output"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = "gemini-3.8-flash"

WEIGHTS = {
    "keywords": 0.7,
    "completeness": 0.3
}


# ==========================================
# 2. GEMINI CLIENT
# ==========================================

def call_llm(prompt):
    """Send a prompt to Gemini and retry temporary server failures."""

    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is missing.")

    client = genai.Client(api_key=API_KEY)

    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            raw = response.text.strip()

            if raw.startswith("```"):
                raw = raw.strip("`").removeprefix("json").strip()

            return json.loads(raw)

        except ServerError as error:
            if error.code != 503 or attempt == max_attempts - 1:
                raise

            delay = min(2 ** (attempt + 1), 30) + random.uniform(0, 1)

            print(
                f"Gemini temporarily unavailable (503). "
                f"Retry {attempt + 1}/{max_attempts - 1} "
                f"in {delay:.1f} seconds..."
            )

            time.sleep(delay)


# ==========================================
# 3. PDF TEXT EXTRACTION
# ==========================================

def extract_text(pdf_path):
    """Extract text from every page of a PDF."""

    pages = []

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    text = "\n".join(pages).strip()

    if not text:
        raise ValueError(
            "No text found. This PDF may require OCR."
        )

    return text


# ==========================================
# 4. RESUME SCHEMA
# ==========================================

SCHEMA = """{
    "name": string,
    "email": string or null,
    "phone": string or null,
    "location": string or null,
    "summary": string or null,
    "skills": [string],
    "experience": [
        {
            "title": string,
            "company": string,
            "dates": string,
            "bullets": [string]
        }
    ],
    "education": [
        {
            "degree": string,
            "school": string,
            "dates": string
        }
    ]
}"""


# ==========================================
# 5. EXTRACTION PROMPT
# ==========================================

EXTRACT_PROMPT = """You are a resume parser.

Convert the resume below into structured JSON.

Return ONLY valid JSON with exactly this shape:

{schema}

Rules:
- Only use information actually present.
- Missing information should be null or [].
- Keep original wording.
- One skill per entry.
- One bullet per entry.

RESUME:
{resume_text}
"""


# ==========================================
# 6. TARGET JOB
# ==========================================

JOB = {
    "title": "Junior Software Engineer",

    "description": """
    We're looking for a Junior Software Engineer
    to build and maintain backend services.

    You will design REST APIs in Python,
    write SQL queries against PostgreSQL,
    collaborate using Git, and package
    services with Docker.

    Experience with unit testing and Agile
    teams is a plus.
    """,

    "keywords": [
        "Python",
        "SQL",
        "REST APIs",
        "Git",
        "Docker",
        "PostgreSQL",
        "unit testing",
        "Agile"
    ]
}


# ==========================================
# 7. TAILORING PROMPT
# ==========================================

TAILOR_PROMPT = """You are an expert resume writer.

Rewrite the candidate's resume for the job below.

Return ONLY valid JSON with exactly this shape:

{schema}

Rules:
- NEVER invent jobs, degrees, dates, or skills.
- Reword bullets only when truthful.
- Use relevant job terminology when accurate.
- Start bullets with strong action verbs.
- Write a two-sentence professional summary.
- Put the most relevant skills first.
- Keep contact information and dates unchanged.

JOB TITLE:
{title}

JOB DESCRIPTION:
{description}

CANDIDATE RESUME:
{resume_json}
"""


# ==========================================
# 8. KEYWORD COVERAGE
# ==========================================

def keyword_coverage(resume, keywords):

    text = json.dumps(resume).lower()

    found = [
        k for k in keywords
        if k.lower() in text
    ]

    missing = [
        k for k in keywords
        if k.lower() not in text
    ]

    coverage = (
        len(found) / len(keywords)
        if keywords else 1.0
    )

    return coverage, found, missing


# ==========================================
# 9. COMPLETENESS
# ==========================================

def completeness(resume):

    checks = {
        "has name": bool(resume.get("name")),
        "has email": bool(resume.get("email")),
        "has phone": bool(resume.get("phone")),
        "has skills section": bool(resume.get("skills")),
        "has experience section": bool(resume.get("experience")),
        "has education section": bool(resume.get("education")),

        "bullets start with a letter": all(
            b[:1].isalpha()
            for job in resume.get("experience", [])
            for b in job.get("bullets", [])
        )
    }

    score = sum(checks.values()) / len(checks)

    return score, checks


# ==========================================
# 10. ATS SCORE
# ==========================================

def ats_score(resume, job):

    kw, found, missing = keyword_coverage(
        resume,
        job["keywords"]
    )

    comp, checks = completeness(resume)

    total = (
        WEIGHTS["keywords"] * kw
        + WEIGHTS["completeness"] * comp
    )

    return {
        "total": round(total * 100, 1),
        "keyword_score": round(kw * 100, 1),
        "completeness_score": round(comp * 100, 1),
        "keywords_missing": missing,

        "failed_checks": [
            name
            for name, ok in checks.items()
            if not ok
        ]
    }


# ==========================================
# 11. PDF STYLES
# ==========================================

NAME = ParagraphStyle(
    "name",
    fontName="Helvetica-Bold",
    fontSize=18,
    spaceAfter=2
)

CONTACT = ParagraphStyle(
    "contact",
    fontName="Helvetica",
    fontSize=10,
    spaceAfter=8
)

HEADING = ParagraphStyle(
    "heading",
    fontName="Helvetica-Bold",
    fontSize=12,
    spaceBefore=10,
    spaceAfter=4
)

BODY = ParagraphStyle(
    "body",
    fontName="Helvetica",
    fontSize=10,
    leading=13
)

ROLE = ParagraphStyle(
    "role",
    fontName="Helvetica-Bold",
    fontSize=10,
    spaceBefore=4
)


# ==========================================
# 12. ESCAPE SPECIAL CHARACTERS
# ==========================================

def esc(text):

    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ==========================================
# 13. PDF GENERATION
# ==========================================

def build_pdf(resume, out_path):

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch
    )

    s = [
        Paragraph(
            esc(resume.get("name", "")),
            NAME
        )
    ]

    # Contact information

    contact = " | ".join(
        esc(resume[k])
        for k in ("email", "phone", "location")
        if resume.get(k)
    )

    s.append(Paragraph(contact, CONTACT))

    # Summary

    if resume.get("summary"):
        s += [
            Paragraph("SUMMARY", HEADING),
            Paragraph(esc(resume["summary"]), BODY)
        ]

    # Experience

    if resume.get("experience"):

        s.append(Paragraph("EXPERIENCE", HEADING))

        for job in resume["experience"]:

            role = (
                f"{esc(job.get('title'))}, "
                f"{esc(job.get('company'))} "
                f"({esc(job.get('dates'))})"
            )

            s.append(Paragraph(role, ROLE))

            bullets = [
                ListItem(Paragraph(esc(b), BODY))
                for b in job.get("bullets", [])
            ]

            if bullets:
                s.append(
                    ListFlowable(
                        bullets,
                        bulletType="bullet",
                        start="-",
                        leftIndent=14
                    )
                )

    # Education

    if resume.get("education"):

        s.append(Paragraph("EDUCATION", HEADING))

        for ed in resume["education"]:

            education = (
                f"{esc(ed.get('degree'))}, "
                f"{esc(ed.get('school'))} "
                f"({esc(ed.get('dates'))})"
            )

            s.append(Paragraph(education, BODY))

    # Skills

    if resume.get("skills"):

        s += [
            Paragraph("SKILLS", HEADING),
            Paragraph(
                esc(", ".join(resume["skills"])),
                BODY
            )
        ]

    # Generate PDF

    doc.build(s)


# ==========================================
# 14. MAIN PROGRAM
# ==========================================

def main():

    if len(sys.argv) != 2:
        print("Usage: python src/main.py <resume.pdf>")
        sys.exit(1)
    resume_path = Path(sys.argv[1]).expanduser().resolve()

    # Create output directory

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Check resume exists

    if not resume_path.exists():
        raise FileNotFoundError(
            f"Resume not found: {resume_path}"
        )

    # Extract text

    print("\nExtracting resume text...")

    resume_text = extract_text(resume_path)

    # Convert resume into JSON

        # Convert resume into JSON (or load previously saved JSON)

    original_json_path = OUTPUT_DIR / f"{resume_path.stem}_original.json"

    if original_json_path.exists():

        print("\nLoading previously parsed resume...")

        with open(original_json_path, "r") as file:
            original = json.load(file)

    else:

        print("\nParsing resume with Gemini...")

        original = call_llm(
            EXTRACT_PROMPT.format(
                schema=SCHEMA,
                resume_text=resume_text
            )
        )

        with open(original_json_path, "w") as file:
            json.dump(original, file, indent=4)

    print("Waiting before the next API request...")
    time.sleep(15)

    # Tailor resume

    print("\nTailoring resume...")

    tailored = call_llm(
        TAILOR_PROMPT.format(
            schema=SCHEMA,
            title=JOB["title"],
            description=JOB["description"],
            resume_json=json.dumps(original, indent=2)
        )
    )

    tailored_json_path = OUTPUT_DIR / "tailored_resume.json"

    with open(tailored_json_path, "w") as file:
        json.dump(tailored, file, indent=4)

    # Calculate scores

    before = ats_score(original, JOB)
    after = ats_score(tailored, JOB)

    print("\n===== ATS RESULTS =====")

    print(f"Before: {before['total']}/100")
    print(f"After:  {after['total']}/100")

    print("\nMissing keywords after tailoring:")

    print(
        ", ".join(after["keywords_missing"])
        or "None"
    )

    # Generate PDF

    output_path = OUTPUT_DIR / "tailored_resume.pdf"

    build_pdf(tailored, output_path)

    print("\nPDF generated successfully!")

    print(f"Saved to: {output_path}")

    # Verify extracted text

    print("\n===== PDF CONTENT =====")

    print(extract_text(output_path))


# ==========================================
# 15. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()