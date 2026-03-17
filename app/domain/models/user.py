import base64
import hashlib
import hmac
import secrets

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.base_model import BaseModel


class User(BaseModel):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    full_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    def set_password(self, password: str):
        iterations = 200_000
        salt = secrets.token_bytes(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii").rstrip("=")
        dk_b64 = base64.urlsafe_b64encode(dk).decode("ascii").rstrip("=")
        self.password_hash = f"pbkdf2_sha256${iterations}${salt_b64}${dk_b64}"

    def check_password(self, password: str) -> bool:
        try:
            scheme, iter_s, salt_b64, dk_b64 = self.password_hash.split("$", 3)
            if scheme != "pbkdf2_sha256":
                return False
            iterations = int(iter_s)
            salt = base64.urlsafe_b64decode(salt_b64 + "==")
            expected = base64.urlsafe_b64decode(dk_b64 + "==")
            dk = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt, iterations
            )
            return hmac.compare_digest(dk, expected)
        except Exception:
            return False

    def __repr__(self):
        return f"<User {self.username}>"
