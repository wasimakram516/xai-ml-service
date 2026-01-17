from fastapi import FastAPI
from app.routers import predict, explain, auth, students, teachers
from app.database import Base, engine

app = FastAPI(
    title="XAI ML Service",
    description="Student At-Risk & Final Prediction API with Explainability",
    version="1.0"
)

# Create DB tables (teachers)
Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return {"message": "XAI ML Service Running"}

# Register routers
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(students.router, prefix="/students", tags=["Students"])
app.include_router(teachers.router, prefix="/teachers", tags=["Teachers"])
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(explain.router, prefix="/explain", tags=["Explainability"])
