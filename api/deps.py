"""依赖注入：请求鉴权。"""
import secrets

from fastapi import Header, HTTPException, status

from doctor_agent import config


async def verify_token(authorization: str | None = Header(default=None)) -> str:
    """校验请求头 ``Authorization: Bearer <token>``。

    校验失败抛 401；成功返回 token 本身（也可用来做后续用户识别）。
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少或格式错误的 Authorization 头",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, config.AUTH_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token 无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
