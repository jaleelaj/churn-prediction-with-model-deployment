from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define a transformer for one-hot encoding
categorical_cols = [1, 2]  # Assuming gender and location are columns 1 and 2
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'  # Pass through the remaining columns as is
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        gender = request.form['gender']
        location = request.form['location']
        subscriptionMonths = float(request.form['subscriptionMonths'])
        monthlyBill = float(request.form['monthlyBill'])
        usageGB = float(request.form['usageGB'])

        # Transform the input data for one-hot encoding
        input_data = np.array([[age, gender, location, subscriptionMonths, monthlyBill, usageGB]])
        input_data_encoded = preprocessor.transform(input_data)

        # Make a prediction using the loaded model
        prediction = model.predict(input_data_encoded)

        # Return the prediction as JSON response
        return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
