# ATS Resume Builder Workshop

A Python application that uses Google's Gemini API
to parse, tailor, score, and generate resumes in PDF format.

## Prerequisites

Before starting, install:

- Python 3.10 or newer
- Git
- A Gemini API key

Get your API key at:
https://aistudio.google.com/apikey

## Installation

### 1. Clone the repository

```bash
git clone <REPO_URL>
```

### 2. Enter the project directory

```bash
cd ats-workshop
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate the environment

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Configure your Gemini API key

Copy the example environment file:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```env
GEMINI_API_KEY=your-api-key
```

Never upload your API key to GitHub.

## Running the Resume Builder

Run:

```bash
python src/main.py samples/resume_clean.pdf
```

The program will:

1. Extract the original resume's text.
2. Convert the text into structured JSON.
3. Tailor the resume to the target job description.
4. Calculate before-and-after ATS scores.
5. Generate a new PDF resume.

Generated files are saved in the `output/` directory.

## Notes

- An internet connection is required for Gemini API requests.
- Gemini may occasionally return temporary 503 errors.
- API usage limits depend on your Google project.
- The ATS score is a custom educational metric.
- Always review generated resumes for accuracy.

### Canva
```bash
https://canva.link/wjyjqndklu7qcsl
```
