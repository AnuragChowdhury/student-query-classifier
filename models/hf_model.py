# models/hf_model.py

from transformers import pipeline

class HuggingFaceClassifier:
    def __init__(self):
        # Using a zero-shot classifier
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # Categories same as your contacts
        self.labels = [
            "Admissions",
            "Examinations",
            "Hostel/Accommodation",
            "Fees & Scholarships",
            "Technical Issues",
            "General Enquiry"
        ]

    def predict(self, text):
        result = self.classifier(text, candidate_labels=self.labels)
        best_idx = result["scores"].index(max(result["scores"]))
        return self.labels[best_idx], round(result["scores"][best_idx] * 100, 2)
