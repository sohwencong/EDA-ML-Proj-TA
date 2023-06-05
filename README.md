## EDA & ML Project (Technical Assessment)

#### a) Problem statement
Objectives: A fishing company will plan its fishing operation for the next day based on the forecasted weather condition. Specifically, the company will not send its fishermen out fishing during a storm to avoid higher maintenance costs. However, if they choose to stay in, due to incorrect forecast, while their competitors went out, they would lose out severely in terms of the size of their catch. Hence, the company has engaged you (an AI Engineer) to build models that make rain prediction for the next day. By the fishing company’s definition, it is said to have rained if there is more than 1.0 mm of rain in the day.<br> 
Evaluate 3 suitable models for predicting whether it will rain the next day.<br><br>
Dataset: The dataset provided contains measurements that the company has collected at four different points in Singapore. Do note that there could be synthetic features in the dataset. Therefore you would need to state and verify any assumptions that you make.<br><br>
Task 1: Exploratory Data Analysis (EDA) using Jupyter Notebook in Python.<br>
Task 2: End-to-end Machine Learning Pipeline (MLP) in Python scripts.

#### b) Overview of the submitted folder and the folder structure:
```
├── .github
│   └── workflows
│       └── github-actions.yml
├── src
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── data_transformation.py
│   └── model_trainer.py
├── .gitignore
├── README.md
├── eda.ipynb
├── requirements.txt
└── run.sh
```
#### c) Instructions for executing the pipeline and modifying any parameters:
1. In source folder, execute 'pip install -r requirements.txt'. This will install all the required packages for execution of the pipeline. 
2. In a git bash terminal, execute './run.sh'. This will execute the pipeline. 
3. For modifying of the models' parameters, you may do so in 'model_trainer.py' file, under 'model_trainer' function.

#### d) Description of logical steps/flow of the pipeline:
1. Reads and loads data into dataframe
2. Preprocessing of data
3. Transformation of data 
4. Machine learning model training, testing and evaluation<br>
    a. Logistic Regression<br>
    b. Random Forest Classifier<br>
    c. Decision Tree Classifier

#### e) Overview of key findings from the EDA conducted in Task 1 and the choices made in the pipeline based on these findings, particularly any feature engineering:
1. Out of 12997 rows of data, 2308 rows were duplicated. They were dropped, keeping the first duplicated row. 11815 rows of data remained. 
2. After preprocessing of data, rows with missing values made up of about 8.2% of the remaining number of rows. This would not have significant effect on the ML model's performance and were removed. 
3. Outliers were present in 'Rainfall', 'Evaporation', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am' and 'Humidity3pm'.
4. Data distribution of 'Rainfall', 'Evaporation', 'WindGustSpeed', 'WindSpeed9am' and 'WindSpeed3pm' were right-skewed; whereas 'Sunshine' and 'Humidity9am' were left-skewed. 

#### f) Describe how the features in the dataset are processed:
1. Preprocessing of data<br>
    a. Drops duplicated rows<br>
    b. Converts negative values to positive in 'Sunshine'<br>
    c. Replaces missing values in 'WindDir9am' and 'WindDir3pm'<br>
    d. Standardize strings in 'Pressure9am' and 'Pressure3pm' to uppercase<br>
    e. Replaces missing values in 'RainToday'<br>
    f. Removes all other rows with missing values<br>
    g. Replaces 'yes' and 'no' with 1 and 0 in 'RainToday' and 'RainTomorrow'
2. Transformation of data 
    a. Outliers treatment using 'Winsorizer'<br>
    b. Splits into training (70%) and testing data (30%)<br>
    c. Numerical features:<br>
        &nbsp;&nbsp;&nbsp;i. Numerical transformation using 'Yeo-Johnson Transformation'<br>
        &nbsp;&nbsp;&nbsp;ii. Feature scaling using 'StandardScaler'<br>
    d. Categorical features:<br>
        &nbsp;&nbsp;&nbsp;i. Encoding using 'OneHotEncoder'<br>
        &nbsp;&nbsp;&nbsp;ii. Feature scaling using 'StandardScaler'

#### g) Explanation of choice of models for each machine learning task:
3 types of classification models were selected, namely Logistic Regression, Random Forest Classifier and Decision Tree Classifier. 
1. Logistic Regression uses a logistic function to come up with the probabilities of possible outcomes. It is designed for classification problems and is very useful in understanding the influence of several independent variables on a single outcome variable. 
2. Random Forest Classifier is a meta-estimator that fits a number of decision trees on various subsamples of datasets and uses the average to improve its predictive accuracy. It reduces over-fitting which makes it more accurate.
3. Decision Tree Classifier produces a sequence of rules that can be used to classify the data. It requires little data preparation and can handle both numerical and categorical data. 

#### h) Evaluation of the models developed:
1. LogisticRegression<br>
F1 Score - 0.69<br>
Accuracy Score - 86.55%<br>
Recall Score - 0.66<br>
Precision Score - 0.72

2. RandomForestClassifier<br>
F1 Score - 0.66<br>
Accuracy Score - 86.19%<br>
Recall Score - 0.59<br>
Precision Score - 0.75

3. DecisionTreeClassifier<br>
F1 Score - 0.57<br>
Accuracy Score - 80.14%<br>
Recall Score - 0.58<br>
Precision Score - 0.56

F1 Score Ranking: ['LogisticRegression', 'RandomForestClassifier', 'DecisionTreeClassifier']<br>
Accuracy Score Ranking: ['LogisticRegression', 'RandomForestClassifier', 'DecisionTreeClassifier']<br>
Recall Score Ranking: ['LogisticRegression', 'RandomForestClassifier', 'DecisionTreeClassifier']<br>
Precision Score Ranking: ['RandomForestClassifier', 'LogisticRegression', 'DecisionTreeClassifier']


#### i) Other considerations for deploying the models developed:
1. The next steps to this problem is to improve on the models' performance by tuning the hyperparameters. 
2. There is an issue of imbalanced data. 23.2% with '1' vs 76.8% with '0' for 'RainTomorrow'. This imbalanced data will cause the models be biased and generate higher probabilities for '0' result.  
