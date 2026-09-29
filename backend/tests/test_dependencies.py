import unittest

from fastapi import HTTPException

from app.dependencies import require_admin
from app.models import UserResponse


class RequireAdminTests(unittest.IsolatedAsyncioTestCase):
    async def test_allows_admin(self) -> None:
        admin = UserResponse(
            id="admin-id",
            email="admin@example.com",
            role="admin",
        )

        result = await require_admin(admin)

        self.assertEqual(result, admin)

    async def test_rejects_regular_user(self) -> None:
        user = UserResponse(
            id="user-id",
            email="user@example.com",
            role="user",
        )

        with self.assertRaises(HTTPException) as context:
            await require_admin(user)

        self.assertEqual(context.exception.status_code, 403)
        self.assertEqual(context.exception.detail, "Administrator access is required.")
