"""
猫语翻译器 - 后端API服务
FastAPI + 猫叫分类模型
"""
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.classify import router as classify_router
from app.api.human_to_cat import router as human2cat_router
import uvicorn

app = FastAPI(
    title="猫语翻译器 API",
    description="猫叫声分类与翻译服务",
    version="0.1.0"
)

# 跨域配置（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务（猫叫音频等）
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 注册路由
app.include_router(classify_router, prefix="/api/v1")
app.include_router(human2cat_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "🐱 猫语翻译器 API 运行中", "version": "0.1.0"}


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
