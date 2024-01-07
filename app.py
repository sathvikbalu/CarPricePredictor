from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your pre-trained machine learning model
model = joblib.load('car_price_predictor')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        features = [float(request.form['feature{}'.format(i)]) for i in range(1, 8)]
        # Convert the input to a numpy array
        input_data = np.array([features])

        # Make a prediction using the loaded model
        prediction = model.predict(input_data)

        # Render the result on the result.html template
        return render_template('result.html', prediction=prediction[0])

#if __name__ == '__main__':
 #   app.run(debug=True)
