from fastapi import FastAPI

from app.schemas import ChargebackRequest, DecisionResponse
from app.service import decide_chargeback
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="Recourse",
    description="AI-assisted chargeback defense decision engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {
        "service": "Recourse",

        "description": "AI-assisted chargeback defense decision engine",

        "status": "ready",

        "docs": "/docs",

        "decision_endpoint": "/decide",

    }
    

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/decide", response_model=DecisionResponse)
def decide(request: ChargebackRequest):
    return decide_chargeback(request)