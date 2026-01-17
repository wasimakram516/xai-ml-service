from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from app.database import SessionLocal
from app.models.teacher import Teacher
from app.security import (
    hash_password,
    verify_password,
    create_access_token
)

router = APIRouter()

# ------------------------------------------------
# DB Dependency
# ------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------------------------------------------
# Schemas
# ------------------------------------------------
class RegisterRequest(BaseModel):
    full_name: str 
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# ------------------------------------------------
# POST /auth/register
# ------------------------------------------------
@router.post("/register")
def register_teacher(
    payload: RegisterRequest,
    db: Session = Depends(get_db)
):
    existing = db.query(Teacher).filter(
        Teacher.email == payload.email
    ).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Teacher already exists"
        )

    teacher = Teacher(
        full_name=payload.full_name,
        email=payload.email,
        hashed_password=hash_password(payload.password)
    )

    db.add(teacher)
    db.commit()

    return {"message": "Teacher registered successfully"}

# ------------------------------------------------
# POST /auth/login
# ------------------------------------------------
@router.post("/login")
def login_teacher(
    payload: LoginRequest,
    db: Session = Depends(get_db)
):
    teacher = db.query(Teacher).filter(
        Teacher.email == payload.email
    ).first()

    if not teacher or not verify_password(
        payload.password, teacher.hashed_password
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_access_token(
        {"sub": teacher.email}
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }
