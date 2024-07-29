from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import pickle
import uvicorn
import nltk
from nltk.corpus import stopwords
import string
from pydantic import BaseModel

# Initialize NLTK and download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

app = FastAPI()
templates = Jinja2Templates(directory='templates')

# Load the models and vectorizer
models = {
    "Naive Bayes": pickle.load(open('Naive Bayes_model.pkl', 'rb')),
    "Logistic Regression": pickle.load(open('Logistic Regression_model.pkl', 'rb')),
    "SVM": pickle.load(open('SVM_model.pkl', 'rb')),
    "Random Forest": pickle.load(open('Random Forest_model.pkl', 'rb'))
}
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predict")
def predict(request: Request,
            description: str = Form(...),
            model_name: str = Form(...)):
    # Preprocess the input
    description = description.lower().translate(str.maketrans('', '', string.punctuation))
    description = ' '.join([word for word in description.split() if word not in stop_words])
    
    # Vectorize the input
    data = vectorizer.transform([description])
    
    # Predict using the selected model
    model = models.get(model_name)
    prediction = model.predict(data)[0]
    
    return templates.TemplateResponse("home.html", {"request": request, "prediction_text": f"Le type d'incident pédit est {prediction}"})

if __name__ == "__main__":
    uvicorn.run(app)
