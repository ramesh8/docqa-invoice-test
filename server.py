import json
import os
from fastapi import FastAPI, UploadFile, File
from dqa import getpredictions, askquestion, FIELDS
import shutil
import time

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "welcome to document ai api"}


@app.get("/questions")
def questions():
    return FIELDS


@app.post("/process")
def process(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        outpf = os.path.join("uploads/", file.filename)
        with open(outpf, "wb") as f:
            shutil.copyfileobj(file.file, f)
        predictions = getpredictions(outpf)
        end_time = time.time()
        return {"predictions": predictions, "time": end_time - start_time}
    except Exception as e:
        return {"error": str(e)}
    finally:
        file.file.close()


@app.post("/process_ask")
def process_ask(qs: str = json.dumps(FIELDS), file: UploadFile = File(...)):
    try:
        start_time = time.time()
        outpf = os.path.join("uploads/", file.filename)
        with open(outpf, "wb") as f:
            shutil.copyfileobj(file.file, f)

        predictions = getpredictions(outpf, fields=json.loads(qs))
        end_time = time.time()
        return {"predictions": predictions, "time": end_time - start_time}
    except Exception as e:
        return {"error": str(e)}
    finally:
        file.file.close()
