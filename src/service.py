#####
## Copyright 2020-present columns.ai
##
## The code belongs to https://columns.ai
## Terms & conditions to be found at `LICENSE.txt`.
##
#####

# ref a blog post for some starter tips
# https://towardsdatascience.com/creating-restful-apis-using-flask-and-python-655bad51b24

# an example to return status code
# return "Would you like some tea?", 418

# add hook for logic executed before each request
# @app.before_request
# def before():
#     print("This is executed BEFORE each request.")

# using flask to serve a basic web service
import logging
from flask import Flask, jsonify, request
from waitress import serve

# import functions to serve api
from model.prophet import demo, predict
from model.bank2 import classify as fcs_v1_2, merchant
from model.bank3 import classify as fcs_v3

# saml model - parse and generation SAML 2.0 specs
# from saml.model import saml_signin, saml_idp_url

app = Flask(__name__)

# config a logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("columns.ai")


###
# prediction API
###

# NOTE: please specify -L option when testing with curl
# for example "curl -L -X POST -H "Content-Type: application/json" -d '{"data": "hello"}' http://localhost:9999/forecast"


@app.route("/hello/", methods=["GET", "POST"])
def hello():
    return jsonify({"content": "Hello Columns!"})


@app.route("/test/", methods=["GET", "POST"])
def test():
    return jsonify(demo())


@app.route("/forecast/", methods=["POST"])
def forecast():
    # see data model ForecastInput is the json object
    return jsonify(predict(request.json))


# categorize API accepts a json array of strings and returns a json array of strings
# test with curl:
# curl -L -X POST -H "Content-Type: application/json" -d '["ACH Debit FLAGSTAR BANK - LOAN PYMT", "Credit Dividend"]' http://localhost:9999/categorize
@app.route("/categorize/", methods=["POST"])
def categorize():
    # see data model ForecastInput is the json object
    return jsonify(fcs_v1_2(request.json, version="v1"))


# categorize API accepts a json array of transaction objects and returns a json array of strings
# test with curl:
# curl -L -X POST -H "Content-Type: application/json" -d '[{"name":"ACH Debit FLAGSTAR BANK - LOAN PYMT"}]' http://localhost:9999/categorize2
@app.route("/categorize2/", methods=["POST"])
def categorize2():
    # see data model ForecastInput is the json object
    return jsonify(fcs_v1_2(request.json, version="v2"))


# categorize API accepts a json array of transaction objects and returns a json array of strings
# test with curl:
# curl -L -X POST -H "Content-Type: application/json" -d '[{"name":"ACH Debit FLAGSTAR BANK - LOAN PYMT"}]' http://localhost:9999/categorize3
@app.route("/categorize3/", methods=["POST"])
def categorize3():
    # see data model ForecastInput is the json object
    return jsonify(fcs_v3(request.json))


# merchant API to predict merchant for given transaction
# curl -L -X POST -H "Content-Type: application/json" -d '["ACH Debit FLAGSTAR BANK - LOAN PYMT"]' http://localhost:9999/merchant
@app.route("/merchant/", methods=["POST"])
def merchant():
    # see data model ForecastInput is the json object
    return jsonify(merchant(request.json))


if __name__ == "__main__":
    log.info("Columns Ai is listening at port 9999")
    serve(app, host="0.0.0.0", port=9999)
