# Harsh Khatri — Portfolio

Modern, responsive AI/ML Engineer + Data Analyst portfolio built with Flask, HTML, CSS and JavaScript.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Add your resume
Place your final PDF at:
`static/assets/Harsh_Khatri_Resume.pdf`

The Resume button will then download it.

## Deploy on Render
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
