from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import text

from auth.security import create_access_token, verify_password
from database import SessionLocal

router = APIRouter(prefix="/auth", tags=["Authentication"])

class LoginRequest(BaseModel):
    username: str
    password: str

def _set_auth_cookie(request: Request, response: Response, token: str) -> None:
    secure = str(getattr(request.url, "scheme", "")).lower() == "https"
    response.set_cookie(
        key="godeyes_access_token", value=token, httponly=True,
        samesite="lax", secure=secure, max_age=3600, path="/"
    )

@router.post("/login")
def login(data: LoginRequest, request: Request, response: Response):
    username = str(data.username or "").strip()
    password = str(data.password or "")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Tên đăng nhập và mật khẩu không được để trống.")

    with SessionLocal() as db:
        main_account = db.execute(text("""
            SELECT id, username, password_hash, role, is_active
            FROM main_accounts
            WHERE LOWER(username) = LOWER(:username)
            LIMIT 1
        """), {"username": username}).mappings().first()

        if main_account is not None:
            if not bool(main_account["is_active"]):
                raise HTTPException(status_code=401, detail="Tài khoản Admin đã bị khóa.")
            if not verify_password(password, str(main_account["password_hash"] or "")):
                raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu.")
            role = str(main_account["role"] or "MAIN_ADMIN").upper()
            if role != "MAIN_ADMIN":
                role = "MAIN_ADMIN"
            token = create_access_token(
                user_id=int(main_account["id"]),
                username=str(main_account["username"]), role=role
            )
            _set_auth_cookie(request, response, token)
            return {"access_token": token, "token_type": "bearer",
                    "username": str(main_account["username"]), "role": role}

        teacher = db.execute(text("""
            SELECT id, username, password_hash, full_name, is_active
            FROM teacher_accounts
            WHERE LOWER(username) = LOWER(:username)
            LIMIT 1
        """), {"username": username}).mappings().first()

        if teacher is None:
            raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu.")
        if not bool(teacher["is_active"]):
            raise HTTPException(status_code=401, detail="Tài khoản giáo viên đã bị khóa.")
        if not verify_password(password, str(teacher["password_hash"] or "")):
            raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu.")

        token = create_access_token(
            user_id=int(teacher["id"]), username=str(teacher["username"]), role="TEACHER"
        )
        _set_auth_cookie(request, response, token)
        return {"access_token": token, "token_type": "bearer",
                "username": str(teacher["username"]), "role": "TEACHER",
                "full_name": str(teacher["full_name"] or "")}

@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="godeyes_access_token", path="/")
    return {"message": "Logged out"}
