"""
猫叫分类 API
接收音频文件，返回分类结果
"""
import os
import uuid
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
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
async def classify_sound(audio: UploadFile = File(...)):
    """
    上传猫叫音频，返回分类结果
    """
    if not audio:
        raise HTTPException(status_code=400, detail="请上传音频文件")
    
    # 验证文件类型
    allowed_types = ["audio/mpeg", "audio/wav", "audio/mp3", "audio/x-m4a", "audio/webm", "audio/ogg"]
    content_type = audio.content_type or ""
    if content_type not in allowed_types:
        # 宽松验证，有些设备content_type不准
        pass
    
    try:
        # 保存上传文件到临时目录
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
        
        return {
            "text": intent_info["text"],
            "emotion": intent_info["emotion"],
            "confidence": int(result.get("confidence", 0) * 100),
            "catSoundUrl": f"/static/audio/cats/{intent_info['cat_sound']}.wav",
            "detail": {
                "intent": intent_key,
                "raw_confidence": result.get("confidence", 0),
                "all_probs": result.get("all_probs", {})
            }
        }
        
    except Exception as e:
        # 确保清理临时文件
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")
