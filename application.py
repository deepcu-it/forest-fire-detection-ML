from flask import Flask, render_template, redirect, request
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app = application

##import ridge and standard scaler
ridge_model = pickle.load(open("models/ridge.pkl",'rb'))
standard_scaler = pickle.load(open("models/scaler.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict",methods=['GET','POST'])
def prediction():
    if request.method == "POST":        
        temperature = float(request.form.get('temperature'))
        rh = float(request.form.get('rh'))
        ws = float(request.form.get('ws'))
        rain = float(request.form.get('rain'))
        ffmc = float(request.form.get('ffmc'))
        dmc = float(request.form.get('dmc'))
        isi = float(request.form.get('isi'))
        bui = float(request.form.get('bui'))
        region = request.form.get('region')

        new_data = standard_scaler.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,bui]])
        result = ridge_model.predict(new_data)
        print(result)
        return render_template("predict.html",results = result[0])
        
    else:
        return render_template("home.html")




if __name__ == "__main__":
    app.run(host="0.0.0.0")