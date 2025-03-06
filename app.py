from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)
# Update CORS to allow requests from your Netlify domain
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",  # Local development
            "https://ai-medical-diagnosis.netlify.app",  # Production Netlify URL
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Get the absolute path to the data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, 'data', 'diseases.csv')

# Global variables to store model and data
model = None
feature_columns = None
le = None

# Load and prepare the data
try:
    print(f"Attempting to load data from: {data_file_path}")
    df = pd.read_csv(data_file_path)
    print("Data loaded successfully")
    print("Columns in dataset:", df.columns.tolist())
    
    # Process symptoms from the dataset
    symptoms_list = []
    for symptoms in df['symptoms'].str.split(','):
        symptoms_list.extend([s.strip().lower() for s in symptoms])
    unique_symptoms = sorted(list(set(symptoms_list)))
    print(f"Number of unique symptoms found: {len(unique_symptoms)}")
    print("Sample symptoms:", unique_symptoms[:10])
    
    # Create feature matrix
    X = pd.DataFrame(0, index=range(len(df)), columns=unique_symptoms)
    for idx, symptoms in enumerate(df['symptoms'].str.split(',')):
        for symptom in symptoms:
            symptom = symptom.strip().lower()
            if symptom in unique_symptoms:
                X.loc[idx, symptom] = 1
    
    # Prepare target variable
    y = df['disease']
    print(f"Number of diseases: {len(y.unique())}")
    
    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    print("Model trained successfully")
    
    # Store the feature columns for prediction
    feature_columns = unique_symptoms
    print(f"Model ready with {len(feature_columns)} symptoms as features")
    
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    print(f"Current working directory: {os.getcwd()}")

@app.route('/health', methods=['GET'])
def health_check():
    if model is None or feature_columns is None or le is None:
        return jsonify({
            "status": "unhealthy",
            "error": "Model not initialized"
        }), 500
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or feature_columns is None or le is None:
            raise ValueError("Model not initialized")
        
        # Get symptoms from request
        data = request.get_json()
        if not data or 'symptoms' not in data:
            raise ValueError("No symptoms provided")
        
        symptoms = [s.strip().lower() for s in data['symptoms']]
        print(f"Received symptoms: {symptoms}")
        
        # Validate symptoms
        if not symptoms:
            raise ValueError("Empty symptoms list")
        
        # Create input data
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        valid_symptoms = []
        invalid_symptoms = []
        
        for symptom in symptoms:
            if symptom in feature_columns:
                input_data[symptom] = 1
                valid_symptoms.append(symptom)
            else:
                invalid_symptoms.append(symptom)
        
        if not valid_symptoms:
            raise ValueError(f"No valid symptoms found. Invalid symptoms: {invalid_symptoms}")
        
        print(f"Valid symptoms: {valid_symptoms}")
        if invalid_symptoms:
            print(f"Invalid symptoms: {invalid_symptoms}")
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_disease = le.inverse_transform(prediction)[0]
        
        # Get probability scores
        probabilities = model.predict_proba(input_data)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_diseases = le.inverse_transform(top_3_indices)
        top_3_probabilities = probabilities[top_3_indices]
        
        # Prepare response
        result = {
            "predicted_disease": predicted_disease,
            "top_3_predictions": [
                {"disease": disease, "probability": float(prob)}
                for disease, prob in zip(top_3_diseases, top_3_probabilities)
            ],
            "valid_symptoms": valid_symptoms,
            "invalid_symptoms": invalid_symptoms
        }
        
        print(f"Prediction result: {result}")
        return jsonify(result), 200
    
    except Exception as e:
        error_message = str(e)
        print(f"Error during prediction: {error_message}")
        return jsonify({"error": error_message}), 400

if __name__ == '__main__':
    app.run(debug=True) 