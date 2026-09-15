"""API 请求 / 响应模型。"""
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, description="用户问题")
    thread_id: str | None = Field(default=None, description="会话 ID；不传则新建会话")
    resume: bool = Field(default=False, description="True = 续写上一轮被截断的回答")

class ChatResponse(BaseModel):
    thread_id: str = Field(description="会话 ID（下次追问要带上它才有记忆）")
    answer: str = Field(description="最终回答")
    query: str = Field(default="", description="改写后的检索查询（调试记忆是否生效）")
    cache_hit: bool = Field(default=False, description="是否命中检索缓存")
    cache_reason: str = Field(default="", description="缓存命中原因")




class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
