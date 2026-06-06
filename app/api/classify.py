"""
猫叫分类 API v5
- v5: f0特征改用IQR过滤后的robust版本，消除pyin八度跳跃错误
- 调试接口返回raw和robust两个版本的特征，方便对比
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
        
        # ★ v5: 返回robust和raw两个版本的关键特征
        features = result.get("features_debug", {})
        response["features"] = {
            "spectral_centroid": features.get("spectral_centroid"),
            "zcr": features.get("zcr"),
            "spectral_flatness": features.get("spectral_flatness"),
            "duration": features.get("duration"),
            "f0_mean": features.get("f0_mean"),           # robust
            "f0_std": features.get("f0_std"),             # robust
            "f0_range": features.get("f0_range"),         # robust
            "f0_median": features.get("f0_median"),       # v5新增
            "voiced_ratio": features.get("voiced_ratio"),
            "high_freq_energy_ratio": features.get("high_freq_energy_ratio"),  # v5新增
            # raw版本对比
            "f0_std_raw": features.get("f0_std_raw"),
            "f0_mean_raw": features.get("f0_mean_raw"),
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
