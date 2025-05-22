import os
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def analyze_resumes(resume_files, job_description):
    """Compares resumes against job description and returns ranked scores."""
    results = []
    jd_embedding = model.encode([job_description])[0]

    for file in resume_files:
        filename = os.path.basename(file)
        text = extract_text_from_pdf(file)

        if not text.strip():
            print(f"{filename}: No text extracted.")
            continue

        resume_embedding = model.encode([text])[0]
        similarity = cosine_similarity([jd_embedding], [resume_embedding])[0][0] * 100
        similarity = round(similarity, 2)

        # Updated Real/Fake logic: only ≤20 is Fake
        ai_status = "Real" if similarity > 45 else "Fake"

        # Mock name/email extraction
        name = filename.replace("_", " ").replace(".pdf", "")
        email = f"{name.replace(' ', '.').lower()}@example.com"

        print(f"{filename}: Score = {similarity:.2f}, Status = {ai_status}")

        results.append({
            "name": name,
            "email": email,
            "file_url": f"uploads/{filename}",
            "score": similarity,
            "ai_status": ai_status
        })

    return results
