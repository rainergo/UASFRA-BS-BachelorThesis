## <ins>Bachelor-Thesis (2021) - B.Sc. Computer Science</ins>
### Title: *[An approach to Out-of-Model feature explanations in the application to credit fraud detection](https://drive.google.com/file/d/1zKTNjlNdYV2ZyAx1voC4LXjQdQtatvsl/view?usp=drive_link)*
#### Author: *Rainer Gogel*
#### Index Terms: *Machine Learning, Transfer Learning, Feature Contributions, Shapley Values/SHAP*

---

### <ins>Background</ins>
This code accompanies my [Bachelor-Thesis](https://drive.google.com/file/d/1zKTNjlNdYV2ZyAx1voC4LXjQdQtatvsl/view?usp=drive_link) (2021) in Computer Science at the 
[Frankfurt University of Applied Sciences](https://www.frankfurt-university.de/en/about-us/faculty-2-computer-science-and-engineering/welcome-to-faculty-2/). 
The Bachelor-Thesis was written in cooperation with the Big Data & Advanced Analytics department 
of [Commerzbank AG](https://www.commerzbank.com/de/hauptnavigation/home/home.html) as part of an effort to port
R code of an existing project to Python. The ongoing project's main task is to detect corporate credit fraud early with the help of Machine Learning (ML) algorithms. 

My Bachelor-Thesis explored the feasibility to extract additional information from new features that cannot be integrated into the
existing ML model (“ORIGINAL MODEL”). The new features are the independent variables of a new (to be trained) ML model (“TRANSFER MODEL”) whose
dependent/target variable are the ranked predictions (percentiles) of the ORIGINAL MODEL. Through these
predictions, the ORIGINAL MODEL and the TRANSFER MODEL are connected. The new TRANSFER MODEL features reveal their information 
when the contributions of the new features to the predictions of the ORIGINAL MODEL are analyzed. This was done by using feature explanation techniques
such as Shapley Values.

The Python scripts and the Jupyter Notebook are simplified for demonstration purposes. The data has been reduced, transformed and anonymized for 
confidentiality reasons.

---

### <ins>General Info</ins>
All functionalities of the Python code can be run in the condensed Jupyter Notebook **"MAIN.ipynb"**. This Notebook comprises 
5 sections:

    I. ORIGINAL MODEL:
            Trains three different estimators 
                - Random Forest
                - Logistic Regression
                - XGBoost
            with the provided training data
                - /data/OriginalModel/Training/orig_learn_data_preprocessed.zip

    II. FEATURE CONTRIBUTION / SHAP VALUES
            Calculates Shapley Values/Contributions of individual features to
            the respective predictions of the ORIGINAL MODEL. Contributions can be:
                - Global: Aggregated feature contributions for the entire dataset 
                    (dataset: train, test or monthly data. This can be set in: config.ini)
                - Local: Feature contributions to the prediction of a single data point 
    
    III. MAKE PREDICTIONS WITH ORIGINAL MODEL
            Makes predictions on unseen data
                - /data/OriginalModel/Monthly/prediction_data.zip
            and calculates percentiles/ranks along a representative quantile portfolio:
                - /data/OriginalModel/Monthly/quantile_portfolio.zip

    IV. TRANSFER MODEL
            Takes the predictions from III. as target/dependent variable to train
            a NEW model (3 estimators) 
                - Random Forest
                - Logistic Regression
                - XGBoost
            with new feature data:
                - /data/TransferModel/Training/zv_1_3_feat_engineered.zip
    
    V. TRANSFER MODEL FEATURE CONTRIBUTIONS / SHAP VALUES
            Calculates Shapley Values/Contributions of the NEW features of the TRANSFER MODEL
            to the respective predictions of the ORIGINAL MODEL. Contributions can be:
                - Global: Aggregated feature contributions for the entire dataset (can be set in: config.ini)
                - Local: Feature contributions to the prediction of a single data point

The **"MAIN.ipynb"** Notebook makes use of Python scripts in the folder: /modelling/*.py

Please find further information to this in the code and in the [Bachelor-Thesis](https://drive.google.com/file/d/1zKTNjlNdYV2ZyAx1voC4LXjQdQtatvsl/view?usp=drive_link).    

---

### <ins>Data</ins>
Data can be found in the folder: /data/*

The data provided comprises:

    1. Training data for the ORIGINAL MODEL:
            - /data/OriginalModel/Training/orig_learn_data_preprocessed.zip
        The (reduced) training data has 14,020 data points with the following features/columns:
            - 'orig_target': The values of the target/dependent variable
            - 3 keys or client identifiers: 'orig_key_1', 'orig_key_2', 'orig_key_3'
            - 664 feature columns: orig_feat_1 through orig_feat_664
        The estimators will be trained with only some of the feature columns, namely those listed in:
            - /configuration/ensemble_index.csv for Random Forest and XGBoost (32 features)
            - /configuration/logit_index.csv for Logistic Regression (99 features)
    
    2. Prediction data (unseen during Training) for the ORIGINAL MODEL:
            - /data/OriginalModel/Monthly/prediction_data.zip
        In order to calculate the values for the "quantile portfolio", additional data is used:
            - /data/OriginalModel/Monthly/quantile_portfolio.zip

    3. Training data for the TRANSFER MODEL:
            - /data/TransferModel/Training/zv_1_3_feat_engineered.zip
        The (reduced) training data has 76,755 data points with 2 key/client-id columns and 293 feature columns.

Please find further information to this in the code and in the [Bachelor-Thesis](https://drive.google.com/file/d/1zKTNjlNdYV2ZyAx1voC4LXjQdQtatvsl/view?usp=drive_link).    


---

### <ins>Setup</ins>
Follow the steps below to run the Jupyter Notebook **"MAIN.ipynb"**.

1. Go to "**/configuration/config.ini**" and change the "*base_path*" in the [DEFAULT] section to the location where you stored these downloaded files.
1. Run the Jupyter Notebook **"MAIN.ipynb"** or sections thereof.

---