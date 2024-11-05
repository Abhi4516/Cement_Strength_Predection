# Cement Strength Prediction Project

### Project Overview

Concrete compressive strength prediction is essential for assessing concrete durability and reliability in construction. This project uses machine learning models to predict compressive strength based on a combination of input variables such as cement composition, additives, aggregate types, and curing age.

### Data Description

#### Input Variables

* **Cement** (kg per m³ mixture)
* **Blast Furnace Slag** (kg per m³ mixture) - Byproduct with silicates and aluminosilicates.
* **Fly Ash** (kg per m³ mixture) - Fine particulate from coal combustion.
* **Water** (kg per m³ mixture)
* **Superplasticizer** (kg per m³ mixture) - Improves workability without excess water.
* **Coarse Aggregate** (kg per m³ mixture)
* **Fine Aggregate** (kg per m³ mixture)
* **Age** (days) - Curing time from 1 to 365 days.

#### Target Variable

* **Concrete Compressive Strength** (MPa)

### Model Training Process

1. **Data Export and Preprocessing** :

* **Imputation** : Missing values are imputed using KNN.
* **Transformation** : Log transformation applied to normalize skewed data.
* **Scaling** : Training and testing data are scaled separately.

1. **Clustering** :

* **KMeans Clustering** : Determines the optimal cluster count using the elbow plot and KneeLocator function.
* **Purpose** : Allows training distinct models per cluster for higher prediction accuracy.

1. **Model Selection** :

* For each cluster, Random Forest Regressor , Linear Regression, XGBoost, Decision Tree are evaluated.
* **Model Evaluation** : R-squared scores determine the best-performing model for each cluster.
* **Model Storage** : Selected models are saved and used for predictions.
