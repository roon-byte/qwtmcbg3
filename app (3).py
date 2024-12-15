
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('knn_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    pregnancies = float(request.form['pregnancies'])
    blood_pressure = float(request.form['blood_pressure'])
    glucose = float(request.form['glucose'])
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    smoking_history = float(request.form['smoking_history'])

    user_input = [[age, pregnancies, blood_pressure, glucose, weight, height, smoking_history]]
    prediction = model.predict(user_input)

    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
