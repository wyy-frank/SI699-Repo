from flask import Flask, render_template, request
from flask import jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from mymodel import FirstModel, SecondModel, ThirdModel
from joblib import dump, load
import json
import pickle

app = Flask(__name__)

data = pd.read_csv('tables/coord_xy_bg.csv')
origin_tracts = data['TRACT'].unique().tolist()[:-1]

age_category = pd.read_csv('tables/pca_values/age_category.csv')
car_share = pd.read_csv('tables/pca_values/car_share.csv')
education = pd.read_csv('tables/pca_values/education.csv')
employment = pd.read_csv('tables/pca_values/employment.csv')
gender = pd.read_csv('tables/pca_values/gender.csv')
hhincome_broad = pd.read_csv('tables/pca_values/hhincome_broad.csv')
license = pd.read_csv('tables/pca_values/license.csv')
numadults = pd.read_csv('tables/pca_values/numadults.csv').astype(str)
numchildren = pd.read_csv('tables/pca_values/numchildren.csv').astype(str)
vehicle_count = pd.read_csv('tables/pca_values/vehicle_count.csv')
cluster = pd.read_csv("tables/clustering.csv", index_col=0)
inter = pd.read_csv("tables/inter_cluster.csv", index_col=0)
outer = pd.read_csv("tables/cross_centroid.csv", index_col=0)
inter_x = pd.read_csv("tables/inter_x.csv", index_col=0)
inter_y = pd.read_csv("tables/inter_y.csv", index_col=0)
outer_x = pd.read_csv("tables/outer_x.csv", index_col=0)
outer_y = pd.read_csv("tables/outer_y.csv", index_col=0)
toy = pd.read_csv("tables/second_toy.csv", index_col=0)
coord = pd.read_csv("tables/coord_xy.csv", index_col=0)
coord_bg = pd.read_csv("tables/coord_xy_bg.csv", index_col=0)
x = pd.read_csv("tables/x.csv", index_col=0)
y = pd.read_csv("tables/y.csv", index_col=0)
w = pd.read_csv("tables/w.csv", index_col=0)


#first_model = FirstModel(cluster)
#first_model.train(x,y,w)
#with open('model/firstmodel.pkl', 'wb') as f:
#  pickle.dump(first_model, f)

second_model = SecondModel(toy, coord, coord_bg)
with open('model/secondmodel.pkl', 'wb') as f:
  pickle.dump(second_model, f)

#third_model = ThirdModel(cluster, inter, outer)
#third_model.train(inter_x, inter_y, outer_x, outer_y)

with open('model/firstmodel.pkl', 'rb') as f:
  first_model = pickle.load(f)

with open('model/secondmodel.pkl', 'rb') as f:
  second_model = pickle.load(f)

with open('model/thirdmodel.pkl', 'rb') as f:
  third_model = pickle.load(f)


# Define routes
@app.route('/', methods=['GET', 'POST'])
def index():
    volume = 0
    prediction_results = ['None','None','None','None','None','None']
    if request.method == 'POST':
        # Get form data
        features = [
            request.form['vehicle_count'],
            request.form['hhincome_broad'],
            request.form['car_share'],
            float(request.form['numadults']),
            float(request.form['numchildren']),
            request.form['age_category'],
            request.form['gender'],
            request.form['employment'],
            request.form['education'],
            request.form['license'],
            float(request.form['origin_census_tract'][:-1]),
        ]

        volume = first_model.predict(features[:11])

        origin_coord = second_model.predict(float(request.form['origin_census_tract']))[0].tolist()[0]

        destination_coord = second_model.predict(float(request.form['origin_census_tract']))[1].tolist()[0]

        destination = second_model.predict(float(request.form['origin_census_tract']))[2]

        # Make prediction using the loaded model)
        features.append(destination)
        prediction = third_model.predict(features)
        prediction[prediction<0] = 0
        prediction[5] = prediction[6]
        prediction = prediction[0:6]

        prediction = np.round(prediction/sum(prediction),2)

        # Assuming prediction is a list of 6 floats
        prediction_results = [str(result) for result in prediction]

        return jsonify(html=render_template('prediction_results.html',
     volume=volume,
     prediction_results=prediction_results),
  origin_coord=origin_coord,
  destination_coord=destination_coord)
        

    # Render the index template with no prediction results
    return render_template('index.html', 
                           age_category=age_category['age_category'].unique().tolist(),
                           car_share=car_share['car_share'].unique().tolist(),
                           education=education['education'].unique().tolist(),
                           employment=employment['employment'].unique().tolist(),
                           gender=gender['gender'].unique().tolist(),
                           hhincome_broad=hhincome_broad['hhincome_broad'].unique().tolist(),
                           license=license['license'].unique().tolist(),
                           numadults=numadults['numadults'].unique().tolist(),
                           numchildren=numchildren['numchildren'].unique().tolist(),
                           vehicle_count=vehicle_count['vehicle_count'].unique().tolist(),
                           origin_tracts=origin_tracts,
                           volume = volume,
                           prediction_results=prediction_results)

if __name__ == '__main__':
    app.run(debug=True)