from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

# Load and prepare the data
try:
    df = pd.read_csv('data/diseases.csv')
    # Prepare features and target
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
except Exception as e:
    print(f"Error during initialization: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get symptoms from request
        data = request.get_json()
        symptoms = data.get('symptoms', {})
        
        # Create a DataFrame with all possible symptoms set to 0
        input_data = pd.DataFrame(0, index=[0], columns=X.columns)
        
        # Set 1 for symptoms that are present
        for symptom in symptoms:
            if symptom in input_data.columns:
                input_data[symptom] = 1
        
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
            ]
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 