from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import UserDB
from ..schemas import UserRegister, UserLogin, UserResponse, AuthResponse
from ..services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Optional[UserDB]:
    """Extracts current user from Bearer token if provided, returns None if not authenticated."""
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None

    payload = decode_access_token(token)
    if not payload:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    return db.query(UserDB).filter(UserDB.id == user_id).first()


def get_current_user(
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
) -> UserDB:
    """Requires authentication. Raises 401 Unauthorized if token is missing or invalid."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in or register.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    """Creates a new user account and returns an auth token."""
    clean_username = payload.username.strip().lower()
    clean_email = payload.email.strip().lower()

    if len(clean_username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters long.")
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")
    if "@" not in clean_email or "." not in clean_email:
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")

    # Check for duplicate username
    if db.query(UserDB).filter(UserDB.username == clean_username).first():
        raise HTTPException(status_code=400, detail=f"Username '{clean_username}' is already taken.")

    # Check for duplicate email
    if db.query(UserDB).filter(UserDB.email == clean_email).first():
        raise HTTPException(status_code=400, detail=f"Email '{clean_email}' is already registered.")

    hashed_pwd = hash_password(payload.password)
    new_user = UserDB(
        username=clean_username,
        email=clean_email,
        hashed_password=hashed_pwd,
        full_name=payload.full_name.strip() if payload.full_name else clean_username.title(),
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(user_id=new_user.id, username=new_user.username)
    return {
        "token": token,
        "user": new_user,
        "message": f"Welcome, {new_user.full_name or new_user.username}! Your account has been created."
    }


@router.post("/login", response_model=AuthResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    """Logs in an existing user with username/email and password."""
    identifier = payload.username.strip().lower()
    user = db.query(UserDB).filter(
        (UserDB.username == identifier) | (UserDB.email == identifier)
    ).first()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password."
        )

    token = create_access_token(user_id=user.id, username=user.username)
    return {
        "token": token,
        "user": user,
        "message": f"Welcome back, {user.full_name or user.username}!"
    }


@router.get("/me", response_model=UserResponse)
def get_me(current_user: UserDB = Depends(get_current_user)):
    """Returns profile for currently authenticated user."""
    return current_user
