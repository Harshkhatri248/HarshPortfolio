from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
import os
import re

app = Flask(__name__)

# Limit uploaded/request payload size to 2 MB
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


PROJECTS = [
    {
        "title": "Machine Learning Classification System",
        "category": "AI / ML",
        "description": "Multi-class prediction system comparing KNN, Naive Bayes, SVM, Random Forest, XGBoost, Gradient Boosting and Neural Networks.",
        "tech": ["Python", "Scikit-learn", "XGBoost", "Pandas"],
        "metric": "94.2% test accuracy",
        "github": "https://github.com/Harshkhatri248"
    },
    {
        "title": "Business Performance Dashboard",
        "category": "Power BI",
        "description": "Interactive KPI dashboard with DAX measures, Power Query transformations, drill-through and self-service analysis.",
        "tech": ["Power BI", "DAX", "SQL", "Power Query"],
        "metric": "45s → 8s refresh time",
        "github": "https://github.com/Harshkhatri248"
    },
    {
        "title": "Sales Analytics Dashboard",
        "category": "Data Analytics",
        "description": "Tableau dashboard exploring sales trends, regional performance, customer segments and product categories with forecasting analysis.",
        "tech": ["Tableau", "SQL", "Excel", "Analytics"],
        "metric": "5 regions · 25+ categories",
        "github": "https://github.com/Harshkhatri248"
    },
      {
        "title": "Face Recognition Login",
        "category": "Computer Vision",
        "description": "Computer-vision authentication prototype using face detection, face encodings and a Flask web interface.",
        "tech": ["Python", "OpenCV", "face_recognition", "Flask"],
        "metric": "Vision-based authentication",
        "github": "https://github.com/Harshkhatri248"
    },
    {
        "title": "Mall Feedback Chatbot",
        "category": "Data Analytics",
        "description": "Streamlit feedback analysis tool using sentiment scoring and interactive visualizations to summarize customer feedback.",
        "tech": ["Python", "Streamlit", "NLTK", "Plotly"],
        "metric": "Interactive sentiment insights",
        "github": "https://github.com/Harshkhatri248"
    }
]


SKILLS = [
    ("Python", 90),
    ("SQL", 88),
    ("Machine Learning", 86),
    ("Power BI / DAX", 88),
    ("Tableau", 82),
    ("Pandas / NumPy", 90),
    ("Flask / FastAPI", 78),
    ("Git / GitHub", 80)
]


@app.route("/")
def home():
    return render_template(
        "index.html",
        projects=PROJECTS,
        skills=SKILLS
    )


@app.post("/api/contact")
def contact():
    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip()
    message = str(data.get("message", "")).strip()

    # Validate required fields
    if not name or not email or not message:
        return jsonify(
            ok=False,
            error="Please complete all required fields."
        ), 400

    # Basic email validation
    email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

    if not re.match(email_pattern, email):
        return jsonify(
            ok=False,
            error="Please enter a valid email address."
        ), 400

    # Store messages locally for development.
    # For production, replace this with a database or email service.
    try:
        os.makedirs("messages", exist_ok=True)

        with open(
            "messages/contact.txt",
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                f"\n--- {datetime.now().isoformat()} ---\n"
                f"Name: {name}\n"
                f"Email: {email}\n"
                f"Message: {message}\n"
            )

    except OSError:
        # Don't expose filesystem errors to visitors
        pass

    return jsonify(
        ok=True,
        message="Thanks! Your message has been received."
    )


@app.get("/resume")
def resume():
    filename = "Harsh_Khatri_Resume.pdf"

    resume_folder = os.path.join(
        app.root_path,
        "static",
        "assets"
    )

    resume_path = os.path.join(
        resume_folder,
        filename
    )

    if not os.path.exists(resume_path):
        return (
            "Resume PDF not added yet. "
            "Put your final PDF at "
            "static/assets/Harsh_Khatri_Resume.pdf",
            404
        )

    return send_from_directory(
        resume_folder,
        filename,
        as_attachment=True
    )


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        service="Harsh Khatri Portfolio"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

