from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)   # 🔥 Important for frontend connection

# Job descriptions
jobs = {
    "Data Scientist": "python machine learning pandas numpy statistics",
    "Web Developer": "html css javascript react flask web development",
    "AI Engineer": "deep learning tensorflow pytorch neural networks python"
}

# Skills database
skills_db = [
    "python","java","machine learning","deep learning",
    "html","css","javascript","react","flask",
    "sql","pandas","numpy","tensorflow"
]

# 📄 Extract text from PDF
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    return text.lower()


# 🧠 Extract skills from text
def extract_skills(text):
    found_skills = []
    for skill in skills_db:
        if skill in text:
            found_skills.append(skill)
    return found_skills


# 📊 Calculate score
def calculate_score(skills):
    return {"total": len(skills) * 10}


# 🤖 Match jobs using ML
def match_jobs(resume_text):
    job_list = list(jobs.values())
    job_titles = list(jobs.keys())

    documents = job_list + [resume_text]

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(documents)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    results = {}
    for i, score in enumerate(similarity[0]):
        results[job_titles[i]] = round(score * 100, 2)

    return results


# 🚀 MAIN API
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["resume"]

    # Extract text
    text = extract_text(file)

    # Process
    skills = extract_skills(text)
    score = calculate_score(skills)
    job_matches = match_jobs(text)

    best_job = max(job_matches, key=job_matches.get)

    # Response
    return jsonify({
        "skills": skills,
        "score": score,
        "job_matches": job_matches,
        "best_match": best_job
    })


# ▶️ Run server
if __name__ == "__main__":
    app.run(debug=True)