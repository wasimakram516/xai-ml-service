from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.models.teacher import Teacher
from app.security import hash_password, verify_password
from app.dependencies.auth import get_current_teacher, get_db

router = APIRouter(prefix="/teachers", tags=["Teachers"])

# ---------------------------
# Schemas
# ---------------------------
class TeacherProfileResponse(BaseModel):
    full_name: str
    email: EmailStr

class TeacherUpdateRequest(BaseModel):
    full_name: str
    email: EmailStr

class PasswordChangeRequest(BaseModel):
    old_password: str = Field(min_length=8, max_length=64)
    new_password: str = Field(min_length=8, max_length=64)

# ---------------------------
# GET /teachers/me
# ---------------------------
@router.get("/me", response_model=TeacherProfileResponse)
def get_profile(
    teacher: Teacher = Depends(get_current_teacher),
):
    return teacher

# ---------------------------
# PUT /teachers/me
# ---------------------------
@router.put("/me", response_model=TeacherProfileResponse)
def update_profile(
    payload: TeacherUpdateRequest,
    teacher: Teacher = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    teacher.full_name = payload.full_name
    teacher.email = payload.email

    db.commit()
    db.refresh(teacher)

    return teacher

# ---------------------------
# PUT /teachers/me/password
# ---------------------------
@router.put("/me/password")
def change_password(
    payload: PasswordChangeRequest,
    teacher: Teacher = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    if not verify_password(
        payload.old_password,
        teacher.hashed_password,
    ):
        raise HTTPException(
            status_code=400,
            detail="Incorrect old password",
        )

    teacher.hashed_password = hash_password(payload.new_password)
    db.commit()

    return {"message": "Password updated successfully"}

# ---------------------------
# DELETE /teachers/me
# ---------------------------
@router.delete("/me")
def delete_account(
    teacher: Teacher = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    db.delete(teacher)
    db.commit()

    return {"message": "Teacher account deleted"}
