from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import joblib
import pdfplumber
import os

app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Student Table
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cgpa = db.Column(db.Float)
    projects = db.Column(db.Integer)
    internships = db.Column(db.Integer)
    certifications = db.Column(db.Integer)
    coding_score = db.Column(db.Integer)
    career = db.Column(db.String(100))
    placement_score = db.Column(db.Float)
    result = db.Column(db.String(100))

# Load ML Model
model = joblib.load("model/placement_model.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    cgpa = float(request.form['cgpa'])
    projects = int(request.form['projects'])
    internships = int(request.form['internships'])
    certifications = int(request.form['certifications'])
    codingscore = int(request.form['codingscore'])
    career = request.form['career']

    # Resume Upload
    resume = request.files['resume']

    filepath = os.path.join("uploads", resume.filename)
    resume.save(filepath)

    # Read Resume PDF
    text = ""

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    # ML Prediction
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

    # Save to Database
    student = Student(
        cgpa=cgpa,
        projects=projects,
        internships=internships,
        certifications=certifications,
        coding_score=codingscore,
        career=career,
        placement_score=score,
        result=result
    )

    db.session.add(student)
    db.session.commit()

    # Career Skills
    required_skills = {
        "AI Engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "SQL"],
        "Data Scientist": ["Python", "Statistics", "Machine Learning", "SQL", "Data Visualization"],
        "Software Engineer": ["Java", "DSA", "OOP", "Git", "SQL"],
        "Data Analyst": ["Excel", "SQL", "Power BI", "Statistics", "Python"]
    }

    skills = required_skills[career]

    # Career Roadmaps
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

    roadmap = roadmaps[career]

    # Resume Skill Detection
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
@app.route('/dashboard')
def dashboard():

    search = request.args.get('search')

    if search:
        students = Student.query.filter(
            Student.career.contains(search)
        ).all()
    else:
        students = Student.query.all()

    return render_template(
        "dashboard.html",
        students=students
    )


# Create Database
with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)