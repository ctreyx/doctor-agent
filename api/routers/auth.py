"""登录鉴权路由。"""
import secrets

from fastapi import APIRouter, HTTPException, status

from doctor_agent import config

from api.schemas import LoginRequest, LoginResponse

router = APIRouter()


@router.post("/auth/login", response_model=LoginResponse, summary="登录获取 token")
async def login(req: LoginRequest):
    """校验账号密码，成功后返回访问 token。"""
    # 用 compare_digest 做常量时间比较，避免时序攻击
    user_ok = secrets.compare_digest(req.username, config.AUTH_USERNAME)
    pwd_ok = secrets.compare_digest(req.password, config.AUTH_PASSWORD)
    if not (user_ok and pwd_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    return LoginResponse(access_token=config.AUTH_TOKEN)
