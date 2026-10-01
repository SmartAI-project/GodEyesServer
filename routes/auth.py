from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import select

from auth.security import create_access_token, verify_password
from database import SessionLocal
from models.account import MainAccount
from models.teacher import TeacherAccount


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(data: LoginRequest, response: Response):

    username = data.username.strip()

    if not username or not data.password:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    with SessionLocal() as db:

        # =========================================================
        # 1. MAIN ACCOUNT
        # =========================================================

        main_account = db.scalar(
            select(MainAccount).where(
                MainAccount.username == username
            )
        )

        if main_account is not None:

            if not main_account.is_active:
                raise HTTPException(
                    status_code=401,
                    detail="Account is disabled"
                )

            if not verify_password(
                data.password,
                main_account.password_hash
            ):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid username or password"
                )

            token = create_access_token(
                user_id=main_account.id,
                username=main_account.username,
                role=main_account.role
            )

            response.set_cookie(
                key="godeyes_access_token",
                value=token,
                httponly=True,
                samesite="lax",
                secure=False,
                max_age=3600,
                path="/"
            )

            return {
                "access_token": token,
                "token_type": "bearer",
                "username": main_account.username,
                "role": main_account.role
            }

        # =========================================================
        # 2. TEACHER ACCOUNT
        # =========================================================

        teacher = db.scalar(
            select(TeacherAccount).where(
                TeacherAccount.username == username
            )
        )

        if teacher is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )

        if not teacher.is_active:
            raise HTTPException(
                status_code=401,
                detail="Account is disabled"
            )

        if not verify_password(
            data.password,
            teacher.password_hash
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )

        token = create_access_token(
            user_id=teacher.id,
            username=teacher.username,
            role="TEACHER"
        )

        response.set_cookie(
            key="godeyes_access_token",
            value=token,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=3600,
            path="/"
        )

        return {
            "access_token": token,
            "token_type": "bearer",
            "username": teacher.username,
            "role": "TEACHER"
        }


@router.post("/logout")
def logout(response: Response):

    response.delete_cookie(
        key="godeyes_access_token",
        path="/"
    )

    return {
        "message": "Logged out"
    }