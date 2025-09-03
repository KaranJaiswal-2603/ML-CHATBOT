import pickle
import os
import pandas as pd
import re
from langdetect import detect
from googletrans import Translator
from gtts import gTTS
import pygame
import speech_recognition as sr

model_path = os.path.join("..", "model2", "Disease_model.pkl")
vectorizer_path = os.path.join("..", "model2", "tfidf_vectorizer.pkl")
dataset_path = os.path.join("..", "model2", "data.csv")

with open(model_path, "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

dataset = pd.read_csv(dataset_path)

translator = Translator()

def speak_text(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        filename = "temp_voice.mp3"
        tts.save(filename)

        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            continue

        pygame.mixer.music.unload()
        os.remove(filename)
    except Exception as e:
        print("[Voice Error]", e)

def voice_input(lang='en-IN'):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak now...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language=lang)
        print(f"You said: {text}")
        return text
    except Exception as e:
        print("Voice input error:", e)
        return None

def predict_disease(symptom_text):
    input_vec = loaded_vectorizer.transform([symptom_text])
    prediction = loaded_model.predict(input_vec)
    return prediction[0]

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'

def translate_text(text, dest_language='en'):
    try:
        if not text or len(text.strip()) < 2:
            return text
        translated = translator.translate(text, dest=dest_language)
        if translated.text is None:
            return text
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def translate_if_needed(text, lang):
    if lang != 'en':
        return translate_text(text, dest_language=lang)
    return text

def get_suggestions(disease, user_lang):
    disease_data = dataset[dataset['Disease'].str.lower() == disease.lower()]
    if disease_data.empty:
        return translate_if_needed("Sorry, no data available for the predicted disease.", user_lang)

    row = disease_data.iloc[0]

    medicines = row['Medicines']
    alternate = row['Alternate Medicines']
    home_remedies = row['Home Remedies']
    emergency = row['Emergency']
    mood = row['Mood Support Message']

    home_remedies_t = translate_text(home_remedies, dest_language=user_lang) if user_lang != 'en' else home_remedies
    emergency_t = translate_text(emergency, dest_language=user_lang) if user_lang != 'en' else emergency
    mood_t = translate_text(mood, dest_language=user_lang) if user_lang != 'en' else mood

    response = f"\nMedicine Suggestions:\n- Regular Medicine: {medicines}\n- Alternate Medicine: {alternate}\n"
    response += f"- Home Remedies: {home_remedies_t}\n"
    response += f"Emergency Instructions: {emergency_t}\n"
    response += f"Mood Support: {mood_t}\n"

    return response

def get_mood_support(user_lang):
    
    mood_input = voice_input(lang=user_lang)  
    if not mood_input:
        mood_input = input("How are you feeling today? (happy, sad, anxious, etc.): ")
    if user_lang != 'en':
        mood_en = translate_text(mood_input, dest_language='en')
    else:
        mood_en = mood_input

    mood_en_lower = mood_en.lower()
    if "sad" in mood_en_lower:
        mood_msg_en = "I'm sorry to hear that. Take deep breaths and stay positive."
    elif "happy" in mood_en_lower:
        mood_msg_en = "That's great! Keep up the positivity."
    elif "anxious" in mood_en_lower or "anxiety" in mood_en_lower:
        mood_msg_en = "Try some relaxation techniques and calm your mind."
    else:
        mood_msg_en = "Thanks for sharing how you feel. Let's proceed."

    return translate_text(mood_msg_en, dest_language=user_lang) if user_lang != 'en' else mood_msg_en

def chatbot():
    print("Welcome to the Disease Prediction Chatbot with Voice & Mood Support!")
    while True:
        try:
            
            print("Please say or type your current mood:")
            user_lang = 'en'  

            mood_text = voice_input(lang='en-IN')
            if not mood_text:
                mood_text = input()
            user_lang = detect_language(mood_text)

            mood_response = get_mood_support(user_lang)
            print(mood_response)
            speak_text(mood_response, user_lang)

            print("Please say or type your symptoms separated by commas:")
            symptoms_text = voice_input(lang=user_lang)
            if not symptoms_text:
                symptoms_text = input()

            user_lang = detect_language(symptoms_text)
            symptoms_en = translate_text(symptoms_text, dest_language='en')

            disease = predict_disease(symptoms_en)

            suggestions = get_suggestions(disease, user_lang)

            print(f"\nPredicted Disease: {disease}")
            print(suggestions)

            speak_text(f"Predicted Disease: {disease}", user_lang)
            speak_text(suggestions, user_lang)

            print("\n-------------------------\n")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    chatbot()
