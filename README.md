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

## Dev
Please refer to `src/model/dev.md` for more details.

## Training Data
To produce training data, we use real transactions in https://www.fina.money. Samples are stored in `src/data/cat_train.csv` for training.
In real model training, we use about 1 million records.
As time goes, we will have more data to train to improve the model's accuracy, also using user feedback to automatically improve the model.

## Structure
This repo is super simple, it is structured as
- data: store training data and trained model
- model: source code to train model and serve model
- service.py: flask service to serve model through REST API.
