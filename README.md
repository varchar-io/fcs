# AI

This is AI module serving Columns intelligence need. Some of external services are directly used by api, such as GPT-3.
But this module includes all the interfaces serving columns internally.

## Flask
Flask is simple service we use build up a web service, install it from
```
pip install Flask
```

## Format
We use `black` for python code formatting. VS code will auto install it and configure it upon the prompt.

## bank.py
This is the trainer to train our transaction classifier. It is a simple model that uses `sklearn` to train a classifier.
The train data is stored in `data/categorizer.csv` and the model is stored in `data/categorizer.joblib`.

## produce training data
To produce training data, we use real transactions fetched from plaid. And then we use below command to produce the training data.
```
# col-7: transaction name, col-8: pfc1, col-9:pfc2 col-10: amount, col-13: merchant
(3.9.12) model %cat clay.csv| cut -d',' -f7,9 | grep -v '"' > ../data/cat_train.csv  
(3.9.12) model %cat shawn.csv| cut -d',' -f7,9 | grep -v '"' >> ../data/cat_train.csv
(3.9.12) model %cat xiao.csv| cut -d',' -f7,9 | grep -v '"' >> ../data/cat_train.csv
```

We can grab as many as possible transactions to train the model. The more the better.
