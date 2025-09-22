import streamlit as st
import pyrebase
import pandas as pd
from datetime import datetime
import json
import altair as alt
import joblib
import importlib

from utils.preprocessing import preprocess_text
import config


from utils.email_sender import send_email

# ---------------- FIREBASE CONFIG ----------------
firebaseConfig = {
    "apiKey": "AIzaSyBDYX3ecGplU3Uq1FYz2YC0LHF6APMca5M",
    "authDomain": "student-email-classifier.firebaseapp.com",
    "databaseURL": "https://student-email-classifier-default-rtdb.asia-southeast1.firebasedatabase.app/",
    "projectId": "student-email-classifier",
    "storageBucket": "student-email-classifier.appspot.com"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

# ---------------- CONTACT DETAILS ----------------
contacts = {
    "Admissions": "admissions@christuniversity.in",
    "Examinations": "exam@christuniversity.in",
    "Hostel/Accommodation": "anuragchowdhury19official@gmail.com",
    "Fees & Scholarships": "scholarship.support@christuniversity.in",
    "Technical Issues": "help@christuniversity.in",
    "General Enquiry": "info@christuniversity.in"
}

# ---------------- LOGGING FUNCTION ----------------
def log_action(user, action):
    with open("logs.txt", "a") as f:
        f.write(f"{datetime.now()} - {user} - {action}\n")

# ---------------- LOAD MODELS ----------------
def load_model():
    if config.MODEL_TYPE == "naive_bayes":
        return joblib.load("classifier.pkl")
    elif config.MODEL_TYPE == "huggingface":
        hf_module = importlib.import_module("models.hf_model")
        return hf_module.HuggingFaceClassifier()
    else:
        raise ValueError("Invalid MODEL_TYPE in config.py")

model = load_model()

# ---------------- UI HEADER ----------------
st.image("assets/university_logo.jpg", use_container_width=True)
st.title("🎓 Query ")

role = st.radio("Select Role:", ["User", "Admin"])

# ---------------- ADMIN FLOW ----------------
if role == "Admin":
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False
        st.session_state.admin_email = None

    if not st.session_state.admin_logged_in:
        st.subheader("🔐 Admin Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login as Admin"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state.admin_logged_in = True
                st.session_state.admin_email = email
                log_action(email, "Admin Logged In")
                st.success("✅ Admin login successful!")
                st.rerun()
            except Exception as e:
                st.error("❌ Invalid Credentials")

    else:
        st.subheader("📊 Admin Panel")
        st.write(f"Welcome, **{st.session_state.admin_email}**")

        # Sidebar menu for navigation
        choice = st.sidebar.radio("Admin Menu", ["Dashboard", "Logs", "Manage Admins", "Model Settings", "Logout"])

        # ---------------- DASHBOARD ----------------
        if choice == "Dashboard":
            st.subheader("📊 Query Classification Dashboard")
            try:
                data = db.child("queries").get().val()
                if not data:
                    st.warning("No queries found yet.")
                else:
                    df = pd.DataFrame(data.values())
                    df["confidence"] = df["confidence"].astype(float)

                    st.subheader("📋 All Queries")
                    st.dataframe(df)

                    st.subheader("📊 Queries by Category")
                    chart1 = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(x="category", y="count()", color="category", tooltip=["category", "count()"])
                    )
                    st.altair_chart(chart1, use_container_width=True)

                    st.subheader("📈 Average Confidence per Category")
                    chart2 = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(x="category", y="mean(confidence)", color="category", tooltip=["category", "mean(confidence)"])
                    )
                    st.altair_chart(chart2, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading dashboard: {e}")

        # ---------------- LOGS ----------------
        elif choice == "Logs":
            st.subheader("📜 System Logs")
            try:
                with open("logs.txt", "r") as f:
                    logs = f.readlines()
                st.text_area("System Logs", "".join(logs), height=200)
            except FileNotFoundError:
                st.warning("No logs found yet.")

        # ---------------- MANAGE ADMINS ----------------
        elif choice == "Manage Admins":
            st.subheader("➕ Add New Admin")
            new_email = st.text_input("New Admin Email")
            new_password = st.text_input("New Admin Password", type="password")
            if st.button("Add Admin"):
                try:
                    auth.create_user_with_email_and_password(new_email, new_password)
                    st.success(f"✅ New admin {new_email} added successfully!")
                    log_action(st.session_state.admin_email, f"Added new admin: {new_email}")
                except Exception as e:
                    st.error(f"❌ Failed to add admin: {e}")

        # ---------------- MODEL SETTINGS ----------------
        elif choice == "Model Settings":
            st.subheader("⚙️ Switch Model")
            chosen_model = st.radio("Select Model", ["naive_bayes", "huggingface"], index=0 if config.MODEL_TYPE=="naive_bayes" else 1)
            if st.button("Update Model"):
                with open("config.py", "w") as f:
                    f.write(f'MODEL_TYPE = "{chosen_model}"\n')
                st.success(f"✅ Model switched to {chosen_model}")
                st.rerun()

        # ---------------- LOGOUT ----------------
        elif choice == "Logout":
            st.session_state.admin_logged_in = False
            st.session_state.admin_email = None
            st.success("✅ Logged out successfully!")
            st.rerun()

# ---------------- USER FLOW ----------------
if role == "User":
    st.subheader("📝 Submit Your Query")

    name = st.text_input("Full Name")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email")
    query = st.text_area("Enter your query")

    if st.button("Classify and Submit"):
        if not name or not phone or not email or not query:
            st.error("⚠️ Please fill all fields.")
        else:
            if config.MODEL_TYPE == "naive_bayes":
                cleaned_query = preprocess_text(query)
                proba = model.predict_proba([cleaned_query])[0]
                categories = model.classes_
                best_idx = proba.argmax()
                category = categories[best_idx]
                confidence = round(proba[best_idx] * 100, 2)
            else:
                category, confidence = model.predict(query)

            recipient = contacts.get(category, "info@christuniversity.in")

            record = {
                "name": name,
                "phone": phone,
                "email": email,
                "query": query,
                "category": category,
                "confidence": confidence,
                "recipient": recipient,
                "timestamp": str(datetime.now())
            }
            db.child("queries").push(record)

            st.success(
                f"✅ Thank you {name}! Your query was classified under **{category}** "
                f"with **{confidence}% confidence**, and forwarded to **{recipient}**."
            )
            log_action(email, f"Submitted query - {query}")


           

            # After saving record to Firebase
            subject_dept = f"New Student Query - {category}"
            body_dept = (
                f"Dear {category} Team,\n\n"
                f"A new query has been submitted by {name} ({email}, {phone}).\n\n"
                f"Query: {query}\n\n"
                f"Predicted Category: {category} ({confidence}%)\n\n"
                f"Please follow up accordingly.\n\n"
                f"Best,\nUniversity Query Classifier"
            )

            # Confirmation email to student
            subject_user = "✅ Your Query Has Been Received"
            body_user = (
                f"Hello {name},\n\n"
                f"Thank you for contacting us. Your query has been classified as **{category}** "
                f"with {confidence}% confidence and forwarded to the concerned department.\n\n"
                f"Our team will get back to you shortly.\n\n"
                f"📌 Query submitted: {query}\n\n"
                f"Best regards,\nUniversity Support Team"
            )

            # Send emails
            dept_status = send_email(recipient, subject_dept, body_dept)
            user_status = send_email(email, subject_user, body_user)

            if dept_status and user_status:
                st.success(f"📧 Email sent to **{recipient}** and confirmation sent to **{email}**")
            elif dept_status:
                st.warning(f"📧 Email sent to **{recipient}**, but confirmation to user failed.")
            elif user_status:
                st.warning(f"📧 Confirmation sent to **{email}**, but department email failed.")
            else:
                st.error("⚠️ Failed to send both emails. Please check SMTP settings.")

