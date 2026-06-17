from flask import Flask, request, render_template
import joblib
import pdfplumber

app = Flask(__name__)

# Load ML Model
model = joblib.load("model/placement_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    cgpa = float(request.form["cgpa"])
    projects = int(request.form["projects"])
    internships = int(request.form["internships"])
    certifications = int(request.form["certifications"])
    codingscore = int(request.form["codingscore"])
    career = request.form["career"]

    resume = request.files["resume"]

    text = ""

    with pdfplumber.open(resume) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    prediction = model.predict([
        [cgpa, projects, internships, certifications, codingscore]
    ])

    probability = model.predict_proba([
        [cgpa, projects, internships, certifications, codingscore]
    ])[0]

    if prediction[0] == 1:
        result = "Likely to Get Placed"
        score = round(probability[1] * 100, 2)
    else:
        result = "Needs Improvement"
        score = round(probability[0] * 100, 2)

    required_skills = {
        "AI Engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "SQL"],
        "Data Scientist": ["Python", "Statistics", "Machine Learning", "SQL", "Data Visualization"],
        "Software Engineer": ["Java", "DSA", "OOP", "Git", "SQL"],
        "Data Analyst": ["Excel", "SQL", "Power BI", "Statistics", "Python"]
    }

    roadmaps = {
        "AI Engineer": [
            "Month 1: Learn Python and SQL",
            "Month 2: Learn Machine Learning",
            "Month 3: Learn Deep Learning",
            "Month 4: Learn TensorFlow",
            "Month 5: Build AI Projects",
            "Month 6: Apply for Internships"
        ],
        "Data Scientist": [
            "Month 1: Learn Python and Statistics",
            "Month 2: Learn SQL",
            "Month 3: Learn Data Analysis",
            "Month 4: Learn Machine Learning",
            "Month 5: Build Data Science Projects",
            "Month 6: Apply for Internships"
        ],
        "Software Engineer": [
            "Month 1: Master Java",
            "Month 2: Learn DSA",
            "Month 3: Learn OOP Concepts",
            "Month 4: Build Web Projects",
            "Month 5: Learn Git and GitHub",
            "Month 6: Apply for Internships"
        ],
        "Data Analyst": [
            "Month 1: Learn Excel",
            "Month 2: Learn SQL",
            "Month 3: Learn Power BI",
            "Month 4: Learn Statistics",
            "Month 5: Build Dashboards",
            "Month 6: Apply for Internships"
        ]
    }

    skills = required_skills[career]
    roadmap = roadmaps[career]

    known_skills = [
        "Python",
        "Java",
        "SQL",
        "Machine Learning",
        "Deep Learning",
        "TensorFlow",
        "Power BI",
        "Statistics",
        "Git",
        "DSA"
    ]

    found_skills = []

    for skill in known_skills:
        if skill.lower() in text.lower():
            found_skills.append(skill)

    missing_skills = []

    for skill in skills:
        if skill not in found_skills:
            missing_skills.append(skill)

    return render_template(
        "result.html",
        result=result,
        score=score,
        career=career,
        found_skills=", ".join(found_skills),
        missing_skills=", ".join(missing_skills),
        roadmap=roadmap
    )


if __name__ == "__main__":
    app.run(debug=True)