from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def main():
    return {"message": "This App to Build a model with Tensorflow and evaluate data"}
