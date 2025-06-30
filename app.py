import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__, static_url_path='/Flask/static')  # Adjust only if static folder is /Flask/static
CORS(app)  # Allow CORS requests from Netlify frontend

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # Get data from form
    Gender = float(request.form['Gender'])
    Hemoglobin = float(request.form['Hemoglobin'])
    MCH = float(request.form['MCH'])
    MCHC = float(request.form['MCHC'])
    MCV = float(request.form['MCV'])

    # Create input array
    features_values = np.array([[Gender, Hemoglobin, MCH, MCHC, MCV]])
    df = pd.DataFrame(features_values, columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])

    # Make prediction
    prediction = model.predict(df)
    result = prediction[0]

    # Create readable message
    if result == 0:
        predi = "You don't have any Anemic Disease"
    else:
        predi = "You have Anemic Disease"

    text = f"Hence, based on calculation: {predi}"
    return render_template("predict.html", prediction_text=text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
