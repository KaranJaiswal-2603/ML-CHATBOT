import pickle

with open("Disease_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

    
def predict_disease(symptom_text):
    input_vec = loaded_vectorizer.transform([symptom_text])
    prediction = loaded_model.predict(input_vec)
    return prediction[0]
def get_user_input_and_predict():
    print("Please enter your symptoms separated by commas:")
    user_input = input()  
    result = predict_disease(user_input)
    print("Predicted Disease:", result)
get_user_input_and_predict()
