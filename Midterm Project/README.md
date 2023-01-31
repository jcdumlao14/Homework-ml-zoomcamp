# PROJECT DESCRIPTION

This project performed as part of the _ML Zoomcamp Course_, Midterm Project. This course is conducted by [Alexey Grigorev](https://bit.ly/3BxeAoB)

## Insurance Claim Prediction
Insurance Claim Prediction help to claim their Health Insurance predicting which customers may be interested in claimed health insurance.

## Dataset Reference:
This Model was built using [“Sample Insurance Claim Prediction Dataset”-kaggle Dataset](https://www.kaggle.com/datasets/easonlai/sample-insurance-claim-prediction-dataset#:~:text=1%2C%20no%3D0-,Insurance,-Usability).

## Data Description
Following is the features used for the prediction model. The _Insuranceclaim_ shows the target value.

| **Features** | **Definitions** |
|---|---|
| age | age of policyholder|
| sex | gender of policy holder (female=0, male=1)|
|bmi | Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 25|
|steps | average walking steps per day of policyholder|
|children| number of children / dependents of policyholder|
|smoker| smoking state of policyholder (non-smoke=0;smoker=1)|
|region| the residential area of policyholder in the US (northeast=0, northwest=1, southeast=2, southwest=3)|
|charges | individual medical costs billed by health insurance|
|insuranceclaim | yes=1, no=0|

This is ["Sample Insurance Claim Prediction Dataset"](https://www.kaggle.com/datasets/easonlai/sample-insurance-claim-prediction-dataset#:~:text=1%2C%20no%3D0-,Insurance,-Usability) which based on "[Medical Cost Personal Datasets][1]" to update sample value on top.

## Features Characteristics
Features characteristics are provided in [DataAnalysis](https://github.com/jcdumlao14/Homework-ml-zoomcamp/blob/main/Midterm%20Project/DataAnalysis.ipynb)

**1. Features Type**
|**Categorical**|**Numerical**|
|---|---|
|sex|age|
|smoker|bmi|
|region|charges|
|children|

**2.Correlated Features**

**_AGE_** had the highest correaltion with **_INSURANCECLAIM_**

**3.High-Risk Features**

**_CHILDREN_, _SMOKER_, _REGION_** had the highest risks, respectively.

**4.Mutual Information**

**_CHILDREN_** had the highest mutual information.

## Evaluation Metrics
**_AUC_ROC_Curve_ and _RMSE_** were used as evaluation Metrics.

## Prediction Model
By evaluating different models, _XGBoost_ achieved the best result.
|**Model**|**RMSE**|**AUC**|
|---|---|---|
|Logistic_Regression|0.3961|0.8536|
|Ridge_Regression|0.4329|0.7870|
|Decision Tree|0.4033|0.8360|
|Random_Forest|0.3680|0.8806|
|***XGBoost***|***0.3734***|***0.9069***|

## Classification Models Used
We will use the following classification models to test our theory of whether the health insurance will be claim or not, given the above fields: 

* [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [Decision Trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/python/index.html)


## FILE DESCRIPTION
Folder Midterm Project includes following files:

|**File Name**|**Description**|
|---|---|
|insurance2.csv|Dataset|
|DataAnalysis.ipynb|Exploratory Data Analysis & Feature important Analysis|
|notebook.ipynb|Data preparation and cleaning & Model selection|
|train.py|Training the final model|
|model.bin|Saved model by pickle|
|predict.py|Loading the model & Serving it via a web service (with Flask)|
|predict_test.py|Testing the model|
|Pipfile & Pipfile.lock|Python virtual environment, Pipenv file|
|Dockerfile|Environment management, Docker, for running file|

[RAW DATA -insurance2.csv](https://raw.githubusercontent.com/jcdumlao14/Homework-ml-zoomcamp/main/insurance2.csv)

[notebook.ipynb](https://github.com/jcdumlao14/Homework-ml-zoomcamp/blob/main/Midterm%20Project/notebook.ipynb)


# RUNNING INSTRUCTION
1. Copy scripts (train, predict and predict_test), pipenv file and Dockerfile to a folder
2. Run Windows Terminal Linux (WSL2) in the that folder
3. Install `pipenv`
   ```
   pip install pipenv
   ```
4. Install essential packages
   ```
   pipenv install numpy pandas scikit-learn==1.0 flask xgboost
   ```
5. Install Docker
 - SingUp for a DockerID in [Docker](https://hub.docker.com/)
 - Download & Intall [Docker Desktop](https://docs.docker.com/desktop/windows/install/)
 
6. In WSL2 run the following command to create the image `midterm-project'
   ```
   docker build -t midterm-project .
   ```
7. Run Docker to loading model
   ```
   docker run -it --rm -p 8888:8888 midterm-project
   ```
8. In another WSL tab run the test 
   ```
   python predict_test.py
   ```
9. The Result would be
   ```
   Customer is interested to Claimed Health Insurance
   ```
