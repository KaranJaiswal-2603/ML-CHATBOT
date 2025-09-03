import pickle
import pandas as pd

with open("Disease_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

dataset_csv_path = "data.csv"  
df = pd.read_csv(dataset_csv_path)

def predict_disease(symptom_text):
    input_vec = loaded_vectorizer.transform([symptom_text])
    prediction = loaded_model.predict(input_vec)
    return prediction[0]

def get_disease_info(disease_name):

    disease_row = df[df['disease'] == disease_name].head(1)
    if not disease_row.empty:
        row = disease_row.iloc[0]
        return {
            "medicine": row.get("medicine", "Not available"),
            "alterrnate_medicine": row.get("alterrnate_medicine", "Not available"),
            "home_remedies": row.get("home_remedies", "Not available"),
            "supportive_message": row.get("supportive_message", "Not available"),
            "emergency": row.get("emergency", "No")
        }
    else:
        return None

def get_user_input_and_predict():
    print("Please enter your symptoms separated by commas:")
    user_input = input()
    predicted_disease = predict_disease(user_input)
    print("\nPredicted Disease:", predicted_disease)

    disease_info = get_disease_info(predicted_disease)
    if disease_info:
        print("Medicine:", disease_info["medicine"])
        print("Alternate Medicine:", disease_info["alterrnate_medicine"])
        print("Home Remedies:", disease_info["home_remedies"])
        print("Supportive Message:", disease_info["supportive_message"])
        print("Emergency Condition:", disease_info["emergency"])
    else:
        print("Additional information for this disease is not available in the dataset.")

if __name__ == "__main__":
    get_user_input_and_predict()
