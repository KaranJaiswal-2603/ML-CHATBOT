import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
import os

# ----------------------------
# Paths and folders
# ----------------------------
model_path = "model2/Disease_model.pkl"
vectorizer_path = "model2/tfidf_vectorizer.pkl"
dataset_path = "model2/data.csv"
users_path = "users.csv"
diary_path = "health_diary.csv"
profile_pics_folder = "profile_pics"

os.makedirs(profile_pics_folder, exist_ok=True)

# ----------------------------
# Load model, vectorizer, dataset
# ----------------------------
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)
with open(vectorizer_path, "rb") as f:
    loaded_vectorizer = pickle.load(f)
dataset = pd.read_csv(dataset_path)

# Ensure CSV files exist with necessary columns
if not os.path.exists(users_path):
    pd.DataFrame(columns=["email", "password", "name", "photo_url"]).to_csv(users_path, index=False)
if not os.path.exists(diary_path):
    pd.DataFrame(columns=["email", "date", "disease", "note"]).to_csv(diary_path, index=False)

users_df = pd.read_csv(users_path)
diary_df = pd.read_csv(diary_path)

if 'disease' not in diary_df.columns:
    diary_df['disease'] = ""

# ----------------------------
# Session State Initialization
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "email" not in st.session_state:
    st.session_state.email = ""
if "name" not in st.session_state:
    st.session_state.name = ""
if "photo_url" not in st.session_state:
    st.session_state.photo_url = ""
if "page" not in st.session_state:
    st.session_state.page = "home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "last_predicted_disease" not in st.session_state:
    st.session_state.last_predicted_disease = ""

# ----------------------------
# Helper Functions
# ----------------------------
def predict_disease(symptom_text):
    input_vec = loaded_vectorizer.transform([symptom_text])
    prediction = loaded_model.predict(input_vec)
    return prediction[0]

def get_suggestions(disease):
    disease_data = dataset[dataset['disease'].str.lower() == disease.lower()]
    if disease_data.empty:
        return "Sorry, no data available for the predicted disease."
    row = disease_data.iloc[0]
    medicines = row.get('medicine', 'N/A')
    alternate = row.get('alterrnate_medicine', 'N/A')
    home_remedies = row.get('home_remedies', 'N/A')
    emergency = row.get('emergency', 'N/A')
    supportive_message = row.get('supportive_message', 'N/A')
    response = f"Medicine: {medicines}\nAlternate Medicine: {alternate}\n"
    response += f"Home Remedies: {home_remedies}\n"
    response += f"Emergency: {emergency}\n"
    response += f"Supportive Message: {supportive_message}"
    return response

def login(email, password):
    global users_df
    user = users_df[(users_df['email'] == email) & (users_df['password'] == password)]
    if not user.empty:
        st.session_state.logged_in = True
        st.session_state.email = email
        # Load user profile info after login
        user_data = user.iloc[0]
        st.session_state.name = user_data.get("name", "")
        st.session_state.photo_url = user_data.get("photo_url", "")
        st.session_state.page = "chatbot"  # redirect logged-in users to chatbot page
        st.success("Login Successful!")
    else:
        st.warning("Invalid email or password!")

def signup(email, password, name=""):
    global users_df
    if email in users_df['email'].values:
        st.warning("Email already registered!")
    else:
        new_user = pd.DataFrame([[email, password, name, ""]], columns=["email", "password", "name", "photo_url"])
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv(users_path, index=False)
        st.success("Signup successful! Please login.")

def logout():
    st.session_state.logged_in = False
    st.session_state.email = ""
    st.session_state.name = ""
    st.session_state.photo_url = ""
    st.session_state.page = "home"
    st.session_state.chat_history = []
    st.session_state.user_input = ""
    st.session_state.last_predicted_disease = ""

# ----------------------------
# Pages
# ----------------------------
def home_page():
    st.title("Welcome to AI Health Assistant")
    st.write("Get started to predict your disease based on symptoms, track your health diary, and get recommendations.")
    if not st.session_state.logged_in:
        if st.button("Get Started"):
            st.session_state.page = "login"
    else:
        if st.button("Go to Chatbot"):
            st.session_state.page = "chatbot"

def login_page():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            login(email, password)
    with col2:
        if st.button("Go to Signup"):
            st.session_state.page = "signup"

def signup_page():
    st.title("Signup")
    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Signup"):
            signup(email, password, name)
    with col2:
        if st.button("Go to Login"):
            st.session_state.page = "login"

def sidebar():
    st.sidebar.title("User Profile")
    st.sidebar.image(st.session_state.photo_url if st.session_state.photo_url else "https://cdn-icons-png.flaticon.com/512/1946/1946429.png", width=80)
    st.sidebar.markdown(f"### {st.session_state.name if st.session_state.name else 'User'}")
    st.sidebar.markdown(st.session_state.email)
    st.sidebar.markdown("---")
    menu = st.sidebar.radio("Navigate", ["Chatbot", "Health Diary", "Profile"])
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        logout()
    return menu

def chatbot_page():
    st.title("Disease Prediction Chatbot")
    chat_container = st.container()
    for sender, msg in st.session_state.chat_history:
        msg_html = msg.replace("\n", "<br>")
        if sender == "user":
            chat_container.markdown(
                f'''<div style="background:#DCF8C6; padding:12px; margin:6px; border-radius:8px; max-width:600px; color:#222;">
                <strong>You:</strong><br>{msg_html}</div>''',
                unsafe_allow_html=True)
        else:
            chat_container.markdown(
                f'''<div style="background:#F8F8F8; padding:12px; margin:6px; border-radius:8px; max-width:600px; color:#222; border:1px solid #eee;">
                <strong>Bot:</strong><br>{msg_html}</div>''',
                unsafe_allow_html=True)

    user_input = st.text_input("Enter your symptoms:", value=st.session_state.user_input)
    st.session_state.user_input = user_input

    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please enter your symptoms.")
        else:
            st.session_state.chat_history.append(("user", user_input))
            disease = predict_disease(user_input)
            st.session_state.last_predicted_disease = disease
            suggestions = get_suggestions(disease)
            advisory = "\n\n⚠️ Please note: This prediction may not be fully accurate. Consult a healthcare professional for a proper diagnosis."
            bot_msg = f"Predicted Disease: {disease}\n{suggestions}{advisory}"
            st.session_state.chat_history.append(("bot", bot_msg))
            st.session_state.user_input = ""

def diary_page():
    global diary_df
    st.title("Health Diary")
    st.subheader(f"Hello, {st.session_state.name if st.session_state.name else st.session_state.email}")

    predicted_disease = st.session_state.get("last_predicted_disease", "")

    symptom_note = st.text_area("Enter your symptoms or health notes:")
    disease_input = st.text_input("Predicted Disease (from chatbot):", value=predicted_disease, disabled=True)

    if st.button("Add Note"):
        if symptom_note.strip():
            new_row = {
                "email": st.session_state.email,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "disease": disease_input,
                "note": symptom_note,
            }
            diary_df = pd.concat([diary_df, pd.DataFrame([new_row])], ignore_index=True)
            diary_df.to_csv(diary_path, index=False)
            st.success("Symptom note added!")
            diary_df = pd.read_csv(diary_path)

    user_notes = diary_df[diary_df['email'] == st.session_state.email]
    st.subheader("Your Health Diary")
    st.table(user_notes[["date", "disease", "note"]])

def profile_page():
    global users_df
    st.title("Profile Settings")

    new_name = st.text_input("Name", st.session_state.name)
    new_email = st.text_input("Email", st.session_state.email)
    photo_upload = st.file_uploader("Upload Profile Picture", type=["png", "jpg", "jpeg"])
    current_photo = st.session_state.photo_url if st.session_state.photo_url else "https://cdn-icons-png.flaticon.com/512/1946/1946429.png"
    st.image(current_photo, width=150)

    if st.button("Save"):
        # Check if new_email already exists for different user
        if new_email != st.session_state.email and new_email in users_df['email'].values:
            st.warning("Email already registered by another user.")
            return

        st.session_state.name = new_name
        old_email = st.session_state.email
        st.session_state.email = new_email

        if photo_upload is not None:
            photo_path = os.path.join(profile_pics_folder, f"{new_email}_photo.png")
            with open(photo_path, "wb") as f:
                f.write(photo_upload.getbuffer())
            st.session_state.photo_url = photo_path

        idxs = users_df.index[users_df['email'] == old_email].tolist()
        if idxs:
            idx = idxs[0]
            users_df.at[idx, 'name'] = new_name
            users_df.at[idx, 'email'] = new_email
            users_df.at[idx, 'photo_url'] = st.session_state.photo_url
        else:
            users_df = pd.concat([users_df, pd.DataFrame([[new_email, "", new_name, st.session_state.photo_url]], columns=users_df.columns)], ignore_index=True)

        users_df.to_csv(users_path, index=False)
        st.success("Profile updated!")

# Main navigation
def main():
    if not st.session_state.logged_in:
        # Show home or login/signup pages
        if st.session_state.page == "login":
            login_page()
        elif st.session_state.page == "signup":
            signup_page()
        else:
            home_page()
    else:
        # Logged-in users see sidebar then selected page
        page = sidebar()
        if page == "Chatbot":
            chatbot_page()
        elif page == "Health Diary":
            diary_page()
        elif page == "Profile":
            profile_page()

if __name__ == "__main__":
    main()
