#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# refer: https://github.com/gerritnowald/budget_book/blob/main/budget_book/import_transactions.ipynb

import numpy as np
import pandas as pd
import json
from json import JSONEncoder

# importing sklearn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# this function trains the model
def train():
    # the training set is just transaction name -> category
    data = pd.read_csv(
        "../data/cat_train.csv",
        encoding="utf-8",
        names=["name", "merchant", "place", "flow", "amount", "category"],
        dtype={
            "name": str,
            "merchant": str,
            "place": str,
            "flow": str,
            "amount": float,
            "category": str,
        },
        keep_default_na=False,
        header=None,
    )

    train_data = data.sample(frac=0.8, random_state=200)
    test_data = data.drop(train_data.index)

    # train the classifier
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)

    # let's build a train column based on name, merchant and amount
    # train_col = train_data['name'] + ' ' +  train_data['merchant'] + ' ' + train_data['amount'].map(lambda x: 'income' if x > 0 else 'expense')
    train_col = train_data["name"] + " " + train_data["merchant"]
    test_col = test_data["name"] + " " + test_data["merchant"]
    # normalize the names to lowercase and remove special characters, and remove words that are less than 3 characters
    train_names = train_col.str.lower().replace("[^a-z]", " ", regex=True)
    test_names = test_col.str.lower().replace("[^a-z]", " ", regex=True)

    train_x = vectorizer.fit_transform(train_names.values.astype("U")).toarray()
    train_y = train_data["category"]
    test_x = vectorizer.transform(test_names.values.astype("U")).toarray()
    test_y = test_data["category"]

    # train the model
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    # classifier = CategoricalNB()
    classifier.fit(train_x, train_y)

    print("train-accuracy: " + str(classifier.score(train_x, train_y)))
    print("test-accuracy:  " + str(classifier.score(test_x, test_y)))

    # save the model
    dump([vectorizer, classifier], "../data/categorizer.joblib")


# this function categorizes a list of transactions
# input is a list of transaction names
loaded = False


def classify(input):
    global loaded, vectorizer, classifier
    if not loaded:
        vectorizer, classifier = load("./src/data/categorizer.joblib")
        loaded = True

    # remove special characters and short words from the input
    input = pd.Series(input).str.lower().replace("[^a-z]", " ", regex=True)
    x = vectorizer.transform(input.values.astype("U")).toarray()
    categories = classifier.predict(x)
    result = json.dumps(categories, cls=NumpyArrayEncoder)
    return result


# if we run this file, we train the model
if __name__ == "__main__":
    train()
    # classify(["wendys"])
