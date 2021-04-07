from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
import re

filename = "model.pkl"
cv = pickle.load(open('transform.pkl',"rb"))
clf = pickle.load(open(filename,"rb"))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form.get("message", False)
        vect = cv.transform([message])
        my_prediction = clf.predict(vect)
    return render_template('result.html',tweet = message,prediction = my_prediction[0])

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)