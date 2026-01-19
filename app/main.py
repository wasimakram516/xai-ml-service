from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, students, teachers
from app.database import Base, engine

app = FastAPI(
    title="XAI ML Service",
    description="Student At-Risk & Final Prediction API with Explainability",
    version="1.0",
)

# ------------------------------------------------------
# CORS
# ------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://xai-dashboard-eight.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# DB init (teachers)
# ------------------------------------------------------
Base.metadata.create_all(bind=engine)

# ------------------------------------------------------
# Health check
# ------------------------------------------------------
@app.get("/")
def root():
    return {"message": "XAI ML Service Running"}

# ------------------------------------------------------
# Routers
# ------------------------------------------------------
app.include_router(auth.router)       # /auth/*
app.include_router(students.router)   # /students/* (protected)
app.include_router(teachers.router)   # /teachers/* (protected)
