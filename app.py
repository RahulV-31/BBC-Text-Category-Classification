from flask import Flask, request, jsonify,render_template
import joblib

# Load models and vectorizer
nb_model = joblib.load('naive_bayes_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Load the homepage

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input text from the form
        input_text = request.form['text']
        
        # Preprocess the input text
        processed_text = " ".join([word.lower() for word in input_text.split()])
        data_tfidf = vectorizer.transform([processed_text])
        
        # Get predictions from models
        nb_prediction = nb_model.predict(data_tfidf)[0]
        lr_prediction = lr_model.predict(data_tfidf)[0]
        
        return render_template('index.html', 
                               input_text=input_text,
                               nb_prediction=nb_prediction,
                               lr_prediction=lr_prediction)

if __name__ == '__main__':
    app.run(debug=True)