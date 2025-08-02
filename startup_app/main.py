import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel

from startup_app.chatbot.chatbot import get_chatbot_answer
from startup_app.valuation_predictor.valuation import predict_valuation, load_artifacts

app = FastAPI()

@app.on_event("startup")
def startup_event():
    load_artifacts()
    
class ValuationRequest(BaseModel):
    amount: float
    growth_rate: float
    industry: str
    country: str
    unit: str = "millions"

@app.post("/predict-valuation")
def get_valuation(data: ValuationRequest):
    try:
        result = predict_valuation(
            amount=data.amount,
            growth_rate=data.growth_rate,
            industry=data.industry,
            country=data.country,
            unit=data.unit
        )
        return {"valuation": result, "unit": data.unit}
    except Exception as e:
        return {"error": str(e)}

class ChatRequest(BaseModel):
    query: str

@app.post("/chatbot-query")
def ask_chatbot(request: ChatRequest):
    response = get_chatbot_answer(request.query)
    return {"response": response}
