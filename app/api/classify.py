"""
猫叫分类 API v4
- 返回值增加 features_debug 字段，方便调试手机录音特征
- v4: classify端点默认返回features字段（不再需要debug参数）
"""
import os
import uuid
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.classifier import CatSoundClassifier

router = APIRouter()

# 初始化分类器（全局单例）
classifier = CatSoundClassifier()

# 猫叫意图映射表
INTENT_MAP = {
    "brushing": {
        "text": "好舒服呀，继续摸我～",
        "emotion": "满足",
        "cat_sound": "purr"
    },
    "food": {
        "text": "我饿了！快给我吃的！",
        "emotion": "急切",
        "cat_sound": "meow_hungry"
    },
    "isolation": {
        "text": "你在哪？别丢下我一个人！",
        "emotion": "焦虑",
        "cat_sound": "meow_anxious"
    },
    "happy": {
        "text": "好开心呀！陪我玩！",
        "emotion": "开心",
        "cat_sound": "meow_happy"
    },
    "angry": {
        "text": "别烦我！我想静静！",
        "emotion": "不满",
        "cat_sound": "hiss"
    },
    "pain": {
        "text": "我不舒服...快帮帮我",
        "emotion": "痛苦",
        "cat_sound": "meow_pain"
    }
}


@router.post("/classify")
async def classify_sound(audio: UploadFile = File(...), debug: bool = Query(False)):
    """
    上传猫叫音频，返回分类结果
    debug=true 时返回原始特征数据
    """
    if not audio:
        raise HTTPException(status_code=400, detail="请上传音频文件")
    
    allowed_types = ["audio/mpeg", "audio/wav", "audio/mp3", "audio/x-m4a", "audio/webm", "audio/ogg"]
    content_type = audio.content_type or ""
    if content_type not in allowed_types:
        pass
    
    try:
        filename = audio.filename or "audio.webm"
        suffix = os.path.splitext(filename)[1]
        if not suffix:
            suffix = '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 调用分类器
        result = classifier.predict(tmp_path)
        
        # 获取意图映射
        intent_key = result.get("intent", "food")
        intent_info = INTENT_MAP.get(intent_key, INTENT_MAP["food"])
        
        # 清理临时文件
        os.unlink(tmp_path)
        
        response = {
            "text": intent_info["text"],
            "emotion": intent_info["emotion"],
            "confidence": int(result.get("confidence", 0) * 100),
            "catSoundUrl": f"/static/audio/cats/{intent_info['cat_sound']}.wav",
            "detail": {
                "intent": intent_key,
                "raw_confidence": result.get("confidence", 0),
                "all_probs": result.get("all_probs", {}),
                "rule_hit": result.get("rule_hit", "")
            }
        }
        
        # 始终返回关键特征（v4改动：不再需要debug参数）
        features = result.get("features_debug", {})
        response["features"] = {
            "spectral_centroid": features.get("spectral_centroid"),
            "zcr": features.get("zcr"),
            "spectral_flatness": features.get("spectral_flatness"),
            "duration": features.get("duration"),
            "f0_mean": features.get("f0_mean"),
            "f0_std": features.get("f0_std"),
            "f0_range": features.get("f0_range"),
            "voiced_ratio": features.get("voiced_ratio"),
        }
        
        if debug:
            response["features_full"] = features
        
        return response
        
    except Exception as e:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")


@router.post("/extract-features")
async def extract_features_endpoint(audio: UploadFile = File(...)):
    """
    提取音频特征（调试用）
    返回原始特征数据，不做分类
    """
    if not audio:
        raise HTTPException(status_code=400, detail="请上传音频文件")
    
    try:
        filename = audio.filename or "audio.webm"
        suffix = os.path.splitext(filename)[1]
        if not suffix:
            suffix = '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        features = classifier.extract_features(tmp_path)
        os.unlink(tmp_path)
        
        return {
            "status": "ok",
            "features": features
        }
        
    except Exception as e:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"特征提取失败: {str(e)}")
