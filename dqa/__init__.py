# simple queries only
import torch
from docquery import pipeline
from docquery.document import load_bytes, load_document, ImageDocument
from docquery.ocr_reader import get_ocr_reader

# from PIL import Image
import os
import re

PIPELINES = {}


def construct_pipeline(task, model):
    global PIPELINES
    if model in PIPELINES:
        return PIPELINES[model]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ret = pipeline(task=task, model=model, device=device)
    PIPELINES[model] = ret
    return ret


def run_pipeline(model, question, document, top_k):
    pipeline = construct_pipeline("document-question-answering", model)
    return pipeline(question=question, **document.context, top_k=top_k)


FIELDS = {
    "Vendor Name": ["Vendor Name - Logo?", "Vendor Name - Address?"],
    "Vendor Address": ["Vendor Address?"],
    "Customer Name": ["Customer Name?"],
    "Customer Address": ["Customer Address?"],
    "Invoice Number": ["Invoice Number?"],
    "Invoice Date": ["Invoice Date?"],
    "Due Date": ["Due Date?"],
    "Subtotal": ["Subtotal?"],
    "Total Tax": ["Total Tax?"],
    "Invoice Total": ["Invoice Total?", "Invoice Amount?"],
    "Amount Due": ["Amount Due?"],
    "Payment Terms": ["Payment Terms?"],
    "Remit To Name": ["Remit To Name?"],
    "Remit To Address": ["Remit To Address?"],
}

alphanumeric = re.compile("([\w\s]+)")
datestr = re.compile("([\w\-\s]+)")  # todo
currencystr = re.compile("[\d\.]+")  # todo
FIELD_TYPES = {
    "Vendor Name": ["str", alphanumeric],
    "Vendor Address": ["str", alphanumeric],
    "Customer Name": ["str", alphanumeric],
    "Customer Address": ["str", alphanumeric],
    "Invoice Number": ["str", alphanumeric],
    "Invoice Date": ["datetime", datestr],
    "Due Date": ["datetime", datestr],
    "Subtotal": ["currency", currencystr],
    "Total Tax": ["currency", currencystr],
    "Invoice Total": ["currency", currencystr],
    "Amount Due": ["currency", currencystr],
    "Payment Terms": ["str", alphanumeric],
    "Remit To Name": ["str", alphanumeric],
    "Remit To Address": ["str", alphanumeric],
}

model = "impira/layoutlm-invoices"


def getpredictions(file_path, details=0, fields=FIELDS, field_patterns=True):
    # file_path = f"files/{fname}.pdf"
    if not os.path.isfile(file_path):
        return {"message": f"File Not found {file_path}"}
    document = load_document(file_path)
    answers = {}
    predictions = []
    for field, questions in fields.items():
        for question in questions:
            prediction = run_pipeline(
                model, question, document, 1
            )  # last argument decides minimum number of predictions
            if details == 0:
                del prediction["word_ids"]

            if field_patterns == True:
                prediction["type_matched"] = bool(
                    re.match(FIELD_TYPES[field][1], prediction["answer"])
                )

            predictions.append(prediction)

        ppr = sorted(predictions, key=lambda d: d["score"], reverse=True)
        answers[field] = ppr[:2]
    return answers


def askquestion(file_path, question):
    # file_path = f"files/{fname}.pdf"
    if not os.path.isfile(file_path):
        return {"message": f"File Not found {file_path}"}
    document = load_document(file_path)
    if question.strip() == "":
        return {"message": "missing question"}
    prediction = run_pipeline(model, question, document, 1)
    return prediction
