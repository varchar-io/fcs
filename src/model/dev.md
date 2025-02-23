Fina Categorization Service
===========================

## Description
We train ML models to categorize transactions. 

## Training data
Please use `data/cat_train.csv` as training data. If moving to cloud storage, we will update here with the new location.

NOTE: we can not open source training data. The file contains small sample data for reference. It makes the code runnable.

## Code and Model
All FCS related code are called bank*.py. The model is trained and stored offline. 
Copy them to data folder before running docker build to make deployable image.

### v1 - bank.py
> This is a naive model trained with sklearn.

V1 model is trained with bank.py and outcome as `data/categorizer.joblib` which is deprecated now.

### v2 - bank2.py
> This is a straightforward model trained with xgboost using sentence transformers for embeddings.

V2 model is trained with bank2.py and outcome as `data/fcs_v2.joblib` which is the current model.

### v3 - bank3.py
> This is fine tuned model on top of an advancecd language model.

V3 model is trained with bank3.py and outcome as `data/fcs_v3` which is going to be adopted.
