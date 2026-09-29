from typing import Literal

from pydantic import BaseModel, EmailStr, Field, SecretStr, field_validator


Role = Literal["user", "admin"]


class GoogleCredentialRequest(BaseModel):
    credential: str = Field(min_length=1)


class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: EmailStr
    password: SecretStr = Field(min_length=8, max_length=128)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        cleaned_name = " ".join(value.split())

        if len(cleaned_name) < 2:
            raise ValueError("Name must contain at least two characters.")

        return cleaned_name

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).lower()


class LoginRequest(BaseModel):
    email: EmailStr
    password: SecretStr = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).lower()


class UserResponse(BaseModel):
    id: str
    email: str
    name: str | None = None
    role: Role


class AuthenticationResponse(BaseModel):
    user: UserResponse
