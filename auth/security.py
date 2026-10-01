import os
import hashlib
import hmac
import base64
from datetime import datetime, timedelta, timezone

from jose import jwt


SECRET_KEY = os.getenv(
    "GODEYES_JWT_SECRET",
    "GodEyes-Local-Secret-Change-Later"
)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# ============================================================
# PASSWORD HASHING
# Compatible with Python 3.14+
# Does not depend on passlib/bcrypt.
# ============================================================

HASH_NAME = "sha256"
ITERATIONS = 600_000
SALT_BYTES = 32


def hash_password(password: str) -> str:
    """
    Create a password hash.

    Format:
        pbkdf2_sha256$iterations$salt$hash
    """

    if not isinstance(password, str):
        raise TypeError("Password must be a string")

    password_bytes = password.encode("utf-8")

    salt = os.urandom(SALT_BYTES)

    derived_key = hashlib.pbkdf2_hmac(
        HASH_NAME,
        password_bytes,
        salt,
        ITERATIONS,
    )

    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii")
    hash_b64 = base64.urlsafe_b64encode(derived_key).decode("ascii")

    return (
        f"pbkdf2_sha256$"
        f"{ITERATIONS}$"
        f"{salt_b64}$"
        f"{hash_b64}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against a stored hash.

    Also attempts to recognize old bcrypt hashes if they
    already exist in the database.
    """

    if not password or not password_hash:
        return False

    # --------------------------------------------------------
    # New PBKDF2 format
    # --------------------------------------------------------

    if password_hash.startswith("pbkdf2_sha256$"):

        try:
            parts = password_hash.split("$")

            if len(parts) != 4:
                return False

            _, iterations_text, salt_b64, hash_b64 = parts

            iterations = int(iterations_text)

            salt = base64.urlsafe_b64decode(
                salt_b64.encode("ascii")
            )

            expected_hash = base64.urlsafe_b64decode(
                hash_b64.encode("ascii")
            )

            actual_hash = hashlib.pbkdf2_hmac(
                HASH_NAME,
                password.encode("utf-8"),
                salt,
                iterations,
            )

            return hmac.compare_digest(
                actual_hash,
                expected_hash
            )

        except Exception:
            return False

    # --------------------------------------------------------
    # Legacy bcrypt support
    # --------------------------------------------------------

    if password_hash.startswith((
        "$2a$",
        "$2b$",
        "$2y$",
    )):

        try:
            import bcrypt

            return bcrypt.checkpw(
                password.encode("utf-8"),
                password_hash.encode("utf-8"),
            )

        except Exception:
            return False

    return False


# ============================================================
# JWT
# ============================================================

def create_access_token(
    user_id: int,
    username: str,
    role: str
) -> str:

    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        ),
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_access_token(token: str) -> dict:

    return jwt.decode(
        token,
        SECRET_KEY,
        algorithms=[ALGORITHM],
    )