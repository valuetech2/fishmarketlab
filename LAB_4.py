from flask import Flask, render_template, request
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load('linear_regression_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = {
        'Length1': float(request.form['Length1']),
        'Length2': float(request.form['Length2']),
        'Length3': float(request.form['Length3']),
        'Height': float(request.form['Height']),
        'Width': float(request.form['Width']),
        'Species_Bream': float(request.form['Species_Bream']),
        'Species_Parkki': float(request.form['Species_Parkki']),
        # Add input fields for other species categories as well
    }

    # Create a DataFrame from the form data
    input_data = pd.DataFrame([data])

    # Make prediction using the loaded model
    prediction = model.predict(input_data)

    # Return the prediction to the webpage
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True,port=8070)
