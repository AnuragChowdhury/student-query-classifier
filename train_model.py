import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from utils.preprocessing import preprocess_text
import nltk
nltk.download('punkt_tab')

# Example dataset (replace with your queries.csv)
data = {
    "text": [
        # Admissions
        "What is the last date to apply for the MSc program?",
        "Can I still submit my admission form after the deadline?",
        "When will the merit list for admissions be published?",
        "Do you accept online applications for undergraduate programs?",
        "What documents are required for the admission process?",

        # Examinations
        "When will the semester exam timetable be released?",
        "Can I apply for revaluation of my answer script?",
        "How do I download my hall ticket for the exams?",
        "What is the passing criteria for the final exams?",
        "When will the exam results be declared?",

        # Hostel / Accommodation
        "Is hostel facility available for first-year students?",
        "How much is the hostel fee per semester?",
        "Can I apply for hostel accommodation online?",
        "Are there separate hostels for boys and girls?",
        "What is the procedure to vacate the hostel mid-semester?",

        # Fees & Scholarships
        "How can I pay my semester fees online?",
        "Is there any late fee if I miss the deadline?",
        "What scholarships are available for international students?",
        "Can I apply for both merit-based and need-based scholarships?",
        "How do I get a refund of excess fee payment?",

        # Technical Issues
        "The university portal is not loading my admission form.",
        "I forgot my password, how do I reset it?",
        "I am unable to upload my documents on the website.",
        "Payment gateway is showing error during fee payment.",
        "My account is locked after multiple login attempts.",

        # General Enquiry
        "Can you share the university’s contact number?",
        "What are the library opening hours?",
        "Does the university have a student counseling cell?",
        "Where can I find the academic calendar for this year?",
        "Is there a helpline for student grievances?"
    ],

    "label": [
        # Admissions
        "Admissions", "Admissions", "Admissions", "Admissions", "Admissions",
        # Examinations
        "Examinations", "Examinations", "Examinations", "Examinations", "Examinations",
        # Hostel
        "Hostel/Accommodation", "Hostel/Accommodation", "Hostel/Accommodation", "Hostel/Accommodation", "Hostel/Accommodation",
        # Fees & Scholarships
        "Fees & Scholarships", "Fees & Scholarships", "Fees & Scholarships", "Fees & Scholarships", "Fees & Scholarships",
        # Technical Issues
        "Technical Issues", "Technical Issues", "Technical Issues", "Technical Issues", "Technical Issues",
        # General Enquiry
        "General Enquiry", "General Enquiry", "General Enquiry", "General Enquiry", "General Enquiry"
    ]
}
df = pd.DataFrame(data)

# Preprocess queries
df["cleaned"] = df["text"].apply(preprocess_text)

# Pipeline: TFIDF + Naive Bayes
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

# Train
model.fit(df["cleaned"], df["label"])

# Save
joblib.dump(model, "classifier.pkl")
print("✅ Model trained and saved as classifier.pkl")
