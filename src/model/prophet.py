#####
## Copyright 2020-present columns.ai
##
## The code belongs to https://columns.ai
## Terms & conditions to be found at `LICENSE.txt`.
##
#####

import logging
from datetime import datetime

import pandas as pd
from prophet import Prophet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("columns.ai")

# used to calculate time
EPORCH = pd.Timestamp("1970-01-01")
ONE_MS = pd.Timedelta("1ms")


class Any(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# load a sample data
def tick(name, span: Any):
    span.end = datetime.timestamp(datetime.now())
    log.info("[%s] time taken: %s", name, span.end - span.start)
    span.start = span.end


def demo():
    span = Any(start=datetime.timestamp(datetime.now()), end=0)
    df = pd.read_csv(
        "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
    )

    # fit historical data
    p = Prophet(
        seasonality_mode="multiplicative",
    )
    tick("make_df", span)
    p.fit(df)
    tick("fit_df", span)

    # predict future
    future = p.make_future_dataframe(periods=30, freq="d", include_history=False)
    tick("make_future", span)
    forecast = p.predict(future)
    tick("predict", span)
    # forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
    # tick("forcast", span)
    return forecast.to_dict(orient="records")


# input is typed as ForecastInput
# expect ForecastOutput as return
# these models are defined in `api/src/shared/shared.ts`
HOUR = 3600
DAY = 24 * HOUR
WEEK = DAY * 7
MONTH = DAY * 30
ERR_RANGE = 0.05


def predict(input):
    if input is None:
        return {"error": "invalid input", "data": []}

    rows = input["data"]
    unit = input["unit"]
    steps = input["count"]

    if (rows is None) or (len(rows) < 2):
        return {"error": "empty input data", "data": []}

    # array of object {time, value}
    numRows = len(rows)

    # number of steps to predict
    # if not specified, predict 20% of historical rows
    if steps is None or steps < 1:
        steps = numRows * 2 / 10

    # seconds of time window
    if unit is None or unit < 2:
        unit = rows[1]["time"] - rows[0]["time"]

    window = None
    if (unit - HOUR) / HOUR < ERR_RANGE:
        window = "H"
    elif (unit - DAY) / DAY < ERR_RANGE:
        window = "D"
    elif (unit - WEEK) / WEEK < ERR_RANGE:
        window = "W"
    elif (unit - MONTH) / MONTH < ERR_RANGE:
        window = "M"

    if window is None:
        return {"error": "time window required", "data": []}

    log.info("Predict: window=%s, steps=%s, rows=%s", window, steps, numRows)
    # let's do the forcasting
    # 1. fit the data into Prophet, interface using time->value, but prophet use ds->y
    # note that, time is in unix time format in seconds (s, ms, ns)
    df = pd.json_normalize(rows)
    df["ds"] = pd.to_datetime(df["time"], unit="ms")
    df.rename(columns={"value": "y"}, inplace=True)

    # 2. create a prophet and fit it with the data frame
    p = Prophet(
        seasonality_mode="multiplicative",
    )
    p.fit(df)

    # 3. predict future
    future = p.make_future_dataframe(periods=steps, freq=window, include_history=False)
    forecast = p.predict(future)

    # convert timestamp to UNIX time
    forecast["time"] = (forecast["ds"] - EPORCH) // ONE_MS
    forecast.rename(
        columns={"yhat": "mean", "yhat_lower": "low", "yhat_upper": "high"},
        inplace=True,
    )

    # 4. convert the result to output and return
    result = forecast[["time", "mean", "low", "high"]]
    return result.to_json(orient="records")
