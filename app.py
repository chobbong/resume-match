import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define functions
def calculate_similarity_scores(jd_text, resume_texts):
    # Combine JD and resume texts
    all_texts = [jd_text] + resume_texts

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Generate TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    return cosine_similarities[0]

# Streamlit UI setup
st.title("Resume Similarity Scoring with Job Description")

# Input for Job Description (JD)
st.header("Enter the Job Description")
job_description = st.text_area("Job Description:", height=200)

# Input for candidate resumes
st.header("Enter Candidate Resumes")
candidate_resumes = []
for i in range(1, 11):
    candidate_resume = st.text_area(f"Candidate {i} Resume:", height=100)
    if candidate_resume:
        candidate_resumes.append(candidate_resume)

# Calculate and display similarity scores if inputs are provided
if st.button("Calculate Similarity Scores"):
    if not job_description:
        st.error("Please enter the Job Description.")
    elif not candidate_resumes:
        st.error("Please enter at least one candidate resume.")
    else:
        # Calculate similarity scores
        similarity_scores = calculate_similarity_scores(job_description, candidate_resumes)

        # Display results
        st.header("Similarity Scores")
        for i, (resume_text, score) in enumerate(zip(candidate_resumes, similarity_scores), 1):
            st.write(f"Candidate {i}: Similarity Score = {score:.4f}")

        # Optional: Sort and display top matches
        top_matches = sorted(zip(range(1, len(candidate_resumes) + 1), similarity_scores), key=lambda x: x[1], reverse=True)
        st.header("Top 5 Matching Candidates")
        for i, (candidate_idx, score) in enumerate(top_matches[:5], 1):
            st.write(f"{i}. Candidate {candidate_idx}: Similarity Score = {score:.4f}")