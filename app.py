import streamlit as st
from openai import OpenAI
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize OpenAI client
def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)

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

def analyze_candidate(client, jd_text, candidate_text):
    prompt = f"Job Description: {jd_text}\n\nCandidate Resume: {candidate_text}\n\n회사의 직무 설명에 언급된 요구 사항(직무 소개, 자격요건 및 우대사항 등을 포함)을 요건별로 분석합니다. 후보자의 Resume 의 내용 중 분석한 회사의 직무 설명에 언급된 직무 요건들이 포함되어 있는지 각 요건별로 자세히 설명합니다. 각 요건별로 자세한 답변을 제공하세요. 답변은 반드시 '한국어'로 해주세요"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096
    )
    return response.choices[0].message.content.strip()

# Streamlit UI setup
st.title("Resume Similarity Scoring with Job Description")

# User input for OpenAI API key
api_key = st.text_input("Enter your OpenAI API key:", type="password")

# Initialize OpenAI client if API key is provided
client = None
if api_key:
    client = initialize_openai_client(api_key)

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

    if st.button(f"Analyze Candidate {i}"):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        elif not job_description:
            st.error("Please enter the Job Description.")
        elif not candidate_resume:
            st.error(f"Please enter the resume for Candidate {i}.")
        else:
            analysis_result = analyze_candidate(client, job_description, candidate_resume)
            st.subheader(f"Candidate {i} Analysis Result")
            st.write(analysis_result)

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
