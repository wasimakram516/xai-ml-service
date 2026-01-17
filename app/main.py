from fastapi import FastAPI
from app.routers import predict, explain

app = FastAPI(
    title="XAI ML Service",
    description="Student At-Risk & Final Prediction API with Explainability",
    version="1.0"
)

@app.get("/")
def root():
    return {"message": "XAI ML Service Running"}

# Register routers
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(explain.router, prefix="/explain", tags=["Explainability"])
