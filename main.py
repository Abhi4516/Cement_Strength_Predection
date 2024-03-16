# app.py

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load models and clustering
with open('XGB_model_cluster_0.pkl', 'rb') as f:
    best_xgb_model_cluster_0 = pickle.load(f)

with open('XGB_model_cluster_1.pkl', 'rb') as f:
    best_xgb_model_cluster_1 = pickle.load(f)

with open('XGB_model_cluster_2.pkl', 'rb') as f:
    best_xgb_model_cluster_2 = pickle.load(f)

with open('XGB_model_cluster_3.pkl', 'rb') as f:
    best_xgb_model_cluster_3 = pickle.load(f)

with open('KMean_cluster.pkl', 'rb') as f:
    kmean_clustering = pickle.load(f)

# Define feature labels
feature_labels = ['Cement', 'Blast Furnace Slag _component_2', 'Fly Ash _component_3', 
                  'Water_component_4', 'Superplasticizer_component_5', 'Coarse Aggregate_component_6', 
                  'Fine Aggregate_component_7', 'Age_day']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_values = []
    for feature_label in feature_labels:
        value = float(request.form[feature_label])
        input_values.append(value)
    
    # Reshape input data
    new_data = np.array(input_values).reshape(1, -1)
    
    # Apply log transformation
    new_data = np.log1p(new_data)
    
    # Predict cluster
    new_data_cluster = kmean_clustering.predict(new_data)
    
    # Predict cement strength based on cluster
    if new_data_cluster == 0:
        prediction = best_xgb_model_cluster_0.predict(new_data)
    elif new_data_cluster == 1:
        prediction = best_xgb_model_cluster_1.predict(new_data)
    elif new_data_cluster == 2:
        prediction = best_xgb_model_cluster_2.predict(new_data)
    elif new_data_cluster == 3:
        prediction = best_xgb_model_cluster_3.predict(new_data)
    else:
        prediction = None
    
    return render_template('predict.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

