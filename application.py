from flask import Flask,request,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

# import ridge regressor and standard scalar pickle
rf_model=pickle.load(open('models/randomforest.pkl','rb'))
standard_scalar=pickle.load(open('models/scalar.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoints():
    if request.method=='POST':
        Cut = request.form.get('cut')
        Color = request.form.get('color')
        Clarity = request.form.get('clarity')
        Carat = float(request.form.get('carat'))
        Depth = float(request.form.get('depth'))
        Table = float(request.form.get('table'))
        x = float(request.form.get('x'))
        y  = float(request.form.get('y'))
        z = float(request.form.get('z'))

        # Manually encode categories (Ensure these match how the model was trained)
        cut_mapping = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
        color_mapping = {"D": 0, "E": 1, "F": 2, "G": 3, "H": 4, "I": 5, "J": 6}
        clarity_mapping = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

        # Convert categories to numeric values
        Cut = cut_mapping.get(Cut, -1)
        Color = color_mapping.get(Color, -1)
        Clarity = clarity_mapping.get(Clarity, -1)

        # If any category is invalid, return an error
        if Cut == -1 or Color == -1 or Clarity == -1:
            return "Invalid input for categorical variable!"

        new_data_scaled = standard_scalar.transform([[Cut,Color,Clarity,Carat,Depth,Table,x,y,z]])
        result = rf_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host='0.0.0.0')

if __name__ == '__main__':
    app.run(debug=True)