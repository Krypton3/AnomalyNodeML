import os
import model
import helper
from fastapi import FastAPI

app = FastAPI()

MODEL_DATA = ['train.csv', 'train_1.csv', 'train_2.csv', 'train_3.csv']
MODEL_LOCATION = "data/anomaly_model.keras"
EVALUATION_LOCATION = "data/evaluate.csv"


@app.get("/")
async def main():
    return {"message": "This App to Build a model with Tensorflow and evaluate data"}


@app.get("/train_model")
async def train_model():
    # find out which data needs to be trained
    if len(MODEL_DATA) == len(helper.TRAINED_DATA):
        return {"status": "error", "message": "No new data is availbale to train a new model."}

    not_trained = ""
    for i in MODEL_DATA:
        if i not in helper.TRAINED_DATA:
            not_trained = i
            break
    model_status = await model.model(not_trained)
    return model_status


@app.get("/clear_model")
async def clear_model():
    if os.path.isfile(MODEL_LOCATION):
        try:
            # Remove the file
            os.remove(MODEL_LOCATION)
            await helper.remove_global()
            return {"status": "success", "message": f"File '{MODEL_LOCATION}' has been removed."}
        except Exception as e:
            return {"status": "error", "message": f"An error occurred while trying to remove the file: {e}"}
    else:
        return {"status": "error", "message": f"File '{MODEL_LOCATION}' does not exist."}


@app.get("/metric")
async def evaluate_model():
    return await model.model_evaluation(MODEL_LOCATION, EVALUATION_LOCATION)
