import pandas as pd
import numpy as np
import xlrd
import re
from fuzzywuzzy.fuzz import ratio
import spacy
import os
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

nlp = spacy.load("en_core_web_sm")

load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

class JobSeekerProfile(BaseModel):
    data: dict

class JobDescription(BaseModel):
    description: dict

# def get_gemini_response(input):
#     model = genai.GenerativeModel('gemini-pro')
#     response = model.generate_content(input)
#     return response.text

collegedunia_data = pd.read_excel('CollegeDuniaRatingss.xls', engine='xlrd')
glassdoor_data = pd.read_excel('GlasdoorInformation.xls', engine='xlrd')

def score_experience(years):
    if years > 10:
        return 100
    elif years > 5:
        return 80
    elif years > 2:
        return 60
    return 40

def get_college_rating(college_name, collegedunia_data):
    max_ratio = 0
    college_rating = 0
    for index, row in collegedunia_data.iterrows():
        data_college_name = row["Title"]
        current_ratio = ratio(str(college_name), str(data_college_name))
        if current_ratio > max_ratio:
            max_ratio = current_ratio
            college_rating = row["Rating"]
    return college_rating*10

def get_company_rating(company_name, glasdoor_data):
    max_ratio = 0
    company_rating = 0
    for index, row in glasdoor_data.iterrows():
        data_company_name = row["Company Name"]
        current_ratio = ratio(str(company_name), str(data_company_name))
        if current_ratio > max_ratio:
            max_ratio = current_ratio
            company_rating = row["Rating"]
    return company_rating*10

def calculate_skill_match(seeker_skills, job_skills):
    total_score = 0
    for job_skill in job_skills:
        max_similarity = max(ratio(job_skill, seeker_skill) for seeker_skill in seeker_skills)
        total_score += max_similarity / 100
    return (total_score / len(job_skills)) * 100 if job_skills else 0

def calculate_notice_period_match(seeker_notice, job_notice):
    seeker_days = int(seeker_notice.split()[0])
    job_days = int(job_notice.split()[0])
    if seeker_days <= job_days:
        return 100
    else:
        return max(0, 100 - (seeker_days - job_days) * 5)

def calculate_location_match(seeker_location, job_location):
    return ratio(seeker_location, job_location)

def get_weights_from_jd(job_description):
    weights = {
    "college": 0.2,
    "experience": 0.2,
    "companies": 0.2,
    "skills": 0.15,
    "certifications": 0.15,
    "projects": 0.1
}
    doc = nlp(job_description)
    college_keywords = ["university", "college", "education", "degree"]
    experience_keywords = ["experience", "years", "skills", "proficient"]
    company_keywords = ["company", "organization", "firm", "industry"]

    def contains_keywords(keywords):
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', job_description, re.IGNORECASE):
                return True
        return False

    for entity in doc.ents:
        if entity.label_ in ["ORG", "GPE"]:
            weights["companies"] += 0.2
        elif entity.label_ == "PERSON":
            weights["experience"] += 0.2
        elif entity.label_ in ["DATE", "TIME"]:
            weights["experience"] += 0.2

    if contains_keywords(college_keywords):
        weights["college"] += 0.2
    if contains_keywords(experience_keywords):
        weights["experience"] += 0.2
    if contains_keywords(company_keywords):
        weights["companies"] += 0.2

    total_weight = sum(weights.values())
    for key, value in weights.items():
        weights[key] = value / total_weight
    return weights

# def extract_skills_certifications_projects(job_description):
#     job_skills = []
#     job_certifications = []
#     job_projects = []

  
#     certification_keywords = ["certification", "certified", "certificate"]
#     project_keywords = ["project", "projects"]
  
#     common_skills = set(["python", "java", "sql", "machine learning", "data analysis", "cloud computing", 
#                          "aws", "azure", "gcp", "javascript", "react", "node.js", "blockchain", "big data",
#                          "full stack development", "mobile app development", "iot"])

#     doc = nlp(job_description)

#     for entity in doc.ents:
#         if entity.label_ == "ORG" or entity.label_ == "PRODUCT":
#             job_skills.append(entity.text)
#         elif any(keyword in entity.text.lower() for keyword in project_keywords):
#             job_projects.append(entity.text)
#         elif any(keyword in entity.text.lower() for keyword in certification_keywords):
#             job_certifications.append(entity.text)

#     for token in doc:
#         token_text_lower = token.text.lower()
#         if token.pos_ == "NOUN" or token.pos_ == "PROPN":
#             if token_text_lower in common_skills:
#                 job_skills.append(token.text)
#             elif token.head.pos_ == "VERB" and token.dep_ in ("dobj", "pobj"):
#                 job_skills.append(token.text)
#         if any(keyword in token_text_lower for keyword in certification_keywords):
#             job_certifications.append(token.text)
#         if any(keyword in token_text_lower for keyword in project_keywords):
#             job_projects.append(token.text)

#     job_skills = list(set(job_skills))
#     job_certifications = list(set(job_certifications))
#     job_projects = list(set(job_projects))

#     return job_skills, job_certifications, job_projects


@app.post("/match_profile/")
async def match_profile(seeker: JobSeekerProfile, job: JobDescription):
    try:
        seeker_info = seeker.data
        job_info = job.description
       
        required_seeker_fields = ["Skills", "Preferred_Location", "Notice_Period", "Education", "Current-Company", "Total-Experience"]
        required_job_fields = ["Required_Skills", "Location", "Required_Notice_Period", "Description"]

        for field in required_seeker_fields:
            if field not in seeker_info:
                raise ValueError(f"Missing required field in seeker profile: {field}")

        for field in required_job_fields:
            if field not in job_info:
                raise ValueError(f"Missing required field in job description: {field}")

        weights = get_weights_from_jd(job_info["Description"])

        skill_match = calculate_skill_match(seeker_info["Skills"].split(','), job_info["Required_Skills"])
        location_match = calculate_location_match(seeker_info["Preferred_Location"], job_info["Location"])
        notice_period_match = calculate_notice_period_match(seeker_info["Notice_Period"], job_info["Required_Notice_Period"])
        
        college_rating = get_college_rating(seeker_info["Education"][0]["Institute"], collegedunia_data)
        company_rating = get_company_rating(seeker_info["Current-Company"], glassdoor_data)
        experience_rating = score_experience(seeker_info["Total-Experience"])

        overall_match = (
            skill_match * weights["skills"] +
            location_match * weights.get("location", 0.1) +  
            notice_period_match * weights.get("notice_period", 0.1) + 
            college_rating * weights["college"] +
            company_rating * weights["companies"] +
            experience_rating * weights["experience"]
        ) / sum(weights.values())  

        response_data = {
            "overall_match": round(overall_match, 2),
            "skill_match": round(skill_match, 2),
            "location_match": round(location_match, 2),
            "notice_period_match": round(notice_period_match, 2),
            "college_rating": round(college_rating, 2),
            "company_rating": round(company_rating, 2),
            "experience_rating": round(experience_rating, 2),
            "weights": weights
        }

        return {"match_results": response_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Job Matching API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

        # input_prompt2 = """
        # You are a skilled ATS (Applicant Tracking System) scanner with expertise in various fields including Data Science,
        # Full Stack Development, Cloud Computing, Data Analysis, Big Data Engineering, IoT, Mobile App Development, Blockchain Development, 
        # and ATS functionality. Your task is to evaluate the resume against the job description provided. 
        # Calculate the percentage match between the resume and the job description based on keyword and skill alignment.
        # Provide the result as a single string in the format: {{"Match":"%"}}.
        # Resume: {resume}
        # Job Description: {jd}
        # """
        # input_filled_prompt2 = input_prompt2.format(resume=resume_info, jd=jd)
        # response2 = get_gemini_response(input_filled_prompt2)

  
    