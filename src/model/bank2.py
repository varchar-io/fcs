#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# refer: https://github.com/gerritnowald/budget_book/blob/main/budget_book/import_transactions.ipynb

import numpy as np
import pandas as pd
import time
import json
import re
from json import JSONEncoder

# importing xgboost
import xgboost as xgb
import torch

# importing models for vectorization
from sentence_transformers import SentenceTransformer
from joblib import dump, load

simple_text = re.compile(r"[^=,a-z]")
condense_text = re.compile(r"\b\w\b")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def map_label(df, mapping):
    df["category"] = df["category"].map(mapping).astype(int)
    return df


def map_label_merchant(df, mapping):
    df["merchant"] = df["merchant"].map(mapping).astype(int)
    return df


def simplify_text(text):
    return condense_text.sub(" ", simple_text.sub(" ", text)).strip()


def concat_text(row):
    name = simplify_text(row["name"].lower())
    merchant = simplify_text(row["merchant"].lower())
    flow = "income" if row["amount"] > 0 else "expense"
    # amount = abs(row["amount"])
    if not name:
        name = "unknown"

    if not merchant:
        merchant = "unknown"

    str = f"name={name}, merchant={merchant}, flow={flow}"
    str = re.sub(r"\s+", " ", str).strip()
    return str


def encode_df(df, vectorizer):
    # use applymap to format the dataframe as a new field
    text = df.apply(lambda row: concat_text(row), axis=1)
    values = text.values.astype("U")
    return vectorizer.encode(values, normalize_embeddings=True)


# encoding method for training merchant prediction
def encode_df_merchant(df, vectorizer):
    # use applymap to format the dataframe as a new field
    text = df.apply(lambda row: simplify_text(row["name"].lower()), axis=1)
    values = text.values.astype("U")
    return vectorizer.encode(values, normalize_embeddings=True)


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

    # filter out category with "." in it - polluted by moneykit
    # data = data[~data["category"].str.endswith("uncategorized")]
    # exclude low frequency categories
    excludes = [
        "uncategorized",
        "investment_buy.buy_to_cover",
        "investment_cash.contribution",
        "investment_cash.tax",
        "investment_fee.adjustment",
        "investment_fee.margin_expense",
        "investment_fee.stock_distribution",
        "investment_sell.exercise",
        "investment_buy.assignment",
        "investment_cash.account_fee",
        "investment_transfer.assignment",
    ]
    data = data[~data["category"].isin(excludes)]

    # build category mapping from enum to int
    unique_categories = data["category"].unique()
    print(f"Unique categories: {unique_categories}")
    num_labels = len(unique_categories)
    mapping = {category: i for i, category in enumerate(unique_categories)}

    # split the data into train data and test data
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)
    print("some training samples:")
    print(train_data.head())
    start_time = time.time()

    # we need to vectorize the data
    # vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), max_features=10)
    # sentence-transformers/all-MiniLM-L6-v2: 99%/75%
    # sentence-transformers/all-MiniLM-L12-v2: 99%/75%
    torch.set_num_threads(1)
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    vectorizer = SentenceTransformer(embedding_model, device="cpu")

    # transform this into xgboost dmatrix by removing the category column
    train_x = train_data.drop("category", axis=1)
    train_y = map_label(train_data[["category"]].copy(), mapping)
    train_vector = encode_df(train_x, vectorizer)

    test_x = test_data.drop("category", axis=1)
    test_y = map_label(test_data[["category"]].copy(), mapping)
    test_vector = encode_df(test_x, vectorizer)

    # learning rate: 0.05, max_depth: 6, n_estimators: 200, 99%/76%
    # tree depth decided by leaves = log(num_labels)
    num_trees = 200
    tree_depth = 8
    print(f"Train xgb: labels={num_labels}, trees={num_trees}, depth={tree_depth}...")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_labels,
        learning_rate=0.05,
        max_depth=tree_depth,
        random_state=42,
        eval_metric=["mlogloss", "merror", "auc"],
        n_estimators=num_trees,
        device="cpu",
    )

    # start training the model
    model.fit(train_vector, train_y)

    # print model accuracy, latest data is 99%/91%
    print("train-time:     " + f"{str(time.time() - start_time)} seconds")
    print("train-accuracy: " + str(model.score(train_vector, train_y)))
    print("test-accuracy:  " + str(model.score(test_vector, test_y)))

    # reverse the mapping because we need to map the enum back to the category
    mapping = {i: category for category, i in mapping.items()}

    # save the model
    dump([vectorizer, mapping, model], "../data/fcs_v2.joblib")


# this function trains the model to predict merchant for given transaction
def train_merchant():
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

    data = data[data["merchant"] != ""]
    data["merchant"] = data["merchant"].str.lower()

    # build category mapping from enum to int
    unique_merchants = data["merchant"].unique()
    num_labels = len(unique_merchants)
    mapping = {merchant: i for i, merchant in enumerate(unique_merchants)}

    # split the data into train data and test data
    train_data = data
    test_data = data.sample(frac=0.1, random_state=42)
    print(f"Rows: {len(data)}, Labels: {num_labels}. Training samples:")
    print(train_data.head())
    start_time = time.time()

    # we need to vectorize the data
    torch.set_num_threads(1)
    embedding_model = "sentence-transformers/all-MiniLM-L12-v2"
    vectorizer = SentenceTransformer(embedding_model, device="cpu")

    # transform this into xgboost dmatrix by removing the category column
    train_x = train_data.drop("merchant", axis=1)
    train_y = map_label_merchant(train_data[["merchant"]].copy(), mapping)
    train_vector = encode_df_merchant(train_x, vectorizer)

    test_x = test_data.drop("merchant", axis=1)
    test_y = map_label_merchant(test_data[["merchant"]].copy(), mapping)
    test_vector = encode_df_merchant(test_x, vectorizer)

    # learning rate: 0.05, max_depth: 6, n_estimators: 200, 99%/76%
    # tree depth decided by leaves = log(num_labels)
    num_trees = 50
    tree_depth = 16
    print(f"Train xgb: labels={num_labels}, trees={num_trees}, depth={tree_depth}...")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_labels,
        learning_rate=0.1,
        max_depth=tree_depth,
        random_state=42,
        eval_metric=["mlogloss", "merror", "auc"],
        n_estimators=num_trees,
        device="cpu",
    )

    # start training the model
    model.fit(train_vector, train_y)

    # print model accuracy, latest data is 99%/91%
    print("train-time:     " + f"{str(time.time() - start_time)} seconds")
    print("train-accuracy: " + str(model.score(train_vector, train_y)))
    print("test-accuracy:  " + str(model.score(test_vector, test_y)))

    # reverse the mapping because we need to map the enum back to the category
    mapping = {i: merchant for merchant, i in mapping.items()}

    # save the model
    dump([vectorizer, mapping, model], "../data/fcs_merchant.joblib")


# this function categorizes a list of transactions
# input is a list of transaction names
loaded = False


# latest model has 91% test accuracy
def classify(input, version="v2"):
    global loaded, vectorizer, mapping, model
    if not loaded:
        vectorizer, mapping, model = load("./src/data/fcs_v2.joblib")
        loaded = True

    # seems there is a segment fault in torch
    torch.set_num_threads(1)

    # if version is v1, the input is a list of transaction names, we need to convert it v2 required object
    if version == "v1":
        input = map(lambda x: {"name": x, "merchant": "", "amount": -1}, input)

    # local manual test: name, merchant and amount are required
    # input = [
    #     {
    #         "name": "REVENUE CONTROL 9800 Airport Blvd",
    #         "merchant": "REVENUE CONTROL",
    #         "amount": -1,
    #     }
    # ]

    # convert input to dataframe
    df = pd.DataFrame(input)

    # add merchant column if it doesn't exist
    if "name" not in df.columns:
        df["name"] = ""

    if "merchant" not in df.columns:
        df["merchant"] = ""

    # add amount column (default to expense) if it doesn't exist
    if "amount" not in df.columns:
        df["amount"] = -1

    # remove special characters and short words from the input
    input = encode_df(df, vectorizer)
    categories = model.predict(input)

    # map the index value in categories to category names using mapping
    categories = list(map(lambda x: mapping[x], categories))
    # print(categories)
    result = json.dumps(categories, cls=NumpyArrayEncoder)
    return result


# api to predict merchant for given list of transaction names
merchant_model_loaded = False


def merchant(input):
    global merchant_model_loaded, vectorizer, mapping, model
    if not merchant_model_loaded:
        vectorizer, mapping, model = load("./src/data/fcs_merchant.joblib")
        merchant_model_loaded = True

    # seems there is a segment fault in torch
    torch.set_num_threads(1)

    # local manual test: name, merchant and amount are required
    # input = ["REVENUE CONTROL 9800 Airport Blvd"]

    # convert input to dataframe
    df = pd.DataFrame({"name": np.array(input)})

    # remove special characters and short words from the input
    input = encode_df_merchant(df, vectorizer)
    merchants = model.predict(input)

    # map the index value in categories to category names using mapping
    merchants = list(map(lambda x: mapping[x], merchants))
    # print(categories)
    result = json.dumps(merchants, cls=NumpyArrayEncoder)
    return result


# if we run this file, we train the model
if __name__ == "__main__":
    # train_merchant()
    train()
    # classify2(None)
