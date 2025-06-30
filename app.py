import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for all domains (Netlify frontend)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return jsonify({"message": "Anemia prediction API is live."})

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get form values
        Gender = float(request.form['Gender'])
        Hemoglobin = float(request.form['Hemoglobin'])
        MCH = float(request.form['MCH'])
        MCHC = float(request.form['MCHC'])
        MCV = float(request.form['MCV'])

        # Prepare input for model
        features = np.array([[Gender, Hemoglobin, MCH, MCHC, MCV]])
        df = pd.DataFrame(features, columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])

        # Predict
        prediction = model.predict(df)
        result = prediction[0]

        if result == 0:
            predi = "You don't have any Anemic Disease."
        else:
            predi = "You have Anemic Disease."

        return jsonify({'result': predi})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
