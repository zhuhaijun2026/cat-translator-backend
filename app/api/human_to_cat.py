"""
人语转猫叫 API
支持两种模式：
1. 文字输入 → 猫叫映射（/human-to-cat）
2. 音频输入 → STT转文字 → 猫叫映射（/human-audio-to-cat）
"""
import os
import uuid
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.services.stt import SpeechToText

router = APIRouter()

# STT实例
stt = SpeechToText()

# 人语意图 → 猫叫映射
HUMAN_TO_CAT_MAP = {
    "饿": {"intent": "food", "cat_sound": "meow_hungry",
           "reply": "🐱 收到！我帮你告诉猫咪～"},
    "吃": {"intent": "food", "cat_sound": "meow_hungry",
           "reply": "🐱 喵～开饭啦！"},
    "饿了吗": {"intent": "food", "cat_sound": "meow_hungry",
              "reply": "🐱 翻译中：你饿不饿？"},
    "过来": {"intent": "happy", "cat_sound": "meow_happy",
            "reply": "🐱 喵～过来呀！"},
    "摸摸": {"intent": "brushing", "cat_sound": "purr",
            "reply": "🐱 呼噜噜～摸摸～"},
    "乖": {"intent": "happy", "cat_sound": "meow_happy",
          "reply": "🐱 喵呜～真乖！"},
    "不行": {"intent": "angry", "cat_sound": "hiss",
            "reply": "🐱 嘶～不可以！"},
    "不要": {"intent": "angry", "cat_sound": "hiss",
            "reply": "🐱 嘶～不准这样！"},
    "爱你": {"intent": "happy", "cat_sound": "meow_happy",
            "reply": "🐱 喵～我也爱你！"},
    "睡觉": {"intent": "brushing", "cat_sound": "purr",
            "reply": "🐱 呼噜噜～晚安～"},
    "无聊": {"intent": "happy", "cat_sound": "meow_happy",
            "reply": "🐱 喵！陪我玩！"},
    "痛": {"intent": "pain", "cat_sound": "meow_pain",
          "reply": "🐱 喵呜...哪里不舒服？"},
    "害怕": {"intent": "isolation", "cat_sound": "meow_anxious",
            "reply": "🐱 喵？别怕，我在～"},
    "你好": {"intent": "happy", "cat_sound": "meow_happy",
            "reply": "🐱 喵～你好呀！"},
    "玩": {"intent": "happy", "cat_sound": "meow_happy",
          "reply": "🐱 喵！一起玩！"},
    "出去": {"intent": "isolation", "cat_sound": "meow_anxious",
            "reply": "🐱 喵？带我出去吗？"},
    "回家": {"intent": "happy", "cat_sound": "meow_happy",
            "reply": "🐱 喵～回家啦！"},
    "洗澡": {"intent": "angry", "cat_sound": "hiss",
            "reply": "🐱 嘶～不要洗澡！"},
    "漂亮": {"intent": "happy", "cat_sound": "meow_happy",
            "reply": "🐱 喵～当然漂亮！"},
    "宝贝": {"intent": "happy", "cat_sound": "meow_happy",
            "reply": "🐱 喵～叫我吗？"},
    "过来": {"intent": "happy", "cat_sound": "meow_happy",
            "reply": "🐱 喵～来啦！"},
    "早安": {"intent": "happy", "cat_sound": "meow_happy",
            "reply": "🐱 喵～早上好！"},
    "晚安": {"intent": "brushing", "cat_sound": "purr",
            "reply": "🐱 呼噜噜～晚安～"},
}

DEFAULT_RESPONSE = {
    "intent": "happy",
    "cat_sound": "meow_happy",
    "reply": "🐱 喵～让我想想怎么翻译..."
}


def match_text_to_cat(text: str) -> dict:
    """根据识别出的文字匹配猫叫意图"""
    text = text.strip().lower()
    for keyword, mapping in HUMAN_TO_CAT_MAP.items():
        if keyword in text:
            return mapping
    return DEFAULT_RESPONSE


class HumanToCatRequest(BaseModel):
    text: str


@router.post("/human-to-cat")
async def human_to_cat(req: HumanToCatRequest):
    """
    人语文字 → 猫叫（纯文字输入）
    """
    text = req.text.strip().lower()
    result = match_text_to_cat(text)
    return {
        "intent": result["intent"],
        "catSoundUrl": f"/static/audio/cats/{result['cat_sound']}.wav",
        "reply": result["reply"]
    }


@router.post("/human-audio-to-cat")
async def human_audio_to_cat(audio: UploadFile = File(...)):
    """
    人语音频 → STT转文字 → 猫叫映射
    
    流程：上传音频 → 语音识别 → 关键词匹配 → 返回猫叫
    """
    if not audio:
        raise HTTPException(status_code=400, detail="请上传音频文件")
    
    tmp_path = None
    try:
        # 保存上传的音频到临时文件
        filename = audio.filename or "audio.webm"
        suffix = os.path.splitext(filename)[1]
        if not suffix:
            suffix = '.webm'
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 第一步：语音识别
        stt_result = stt.recognize(tmp_path, language="zh-CN")
        recognized_text = stt_result.get("text", "")
        stt_confidence = stt_result.get("confidence", 0.0)
        
        # 第二步：关键词匹配
        if recognized_text:
            cat_result = match_text_to_cat(recognized_text)
        else:
            cat_result = DEFAULT_RESPONSE
            recognized_text = "(未识别到语音内容)"
        
        return {
            "text": recognized_text,
            "emotion": _intent_to_emotion(cat_result["intent"]),
            "confidence": int(stt_confidence * 100),
            "catSoundUrl": (
                f"/static/audio/cats/{cat_result['cat_sound']}.wav"
            ),
            "reply": cat_result["reply"],
            "intent": cat_result["intent"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"人语翻译失败: {str(e)}"
        )
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except:
                pass


def _intent_to_emotion(intent: str) -> str:
    """意图转情绪标签"""
    emotion_map = {
        "food": "急切",
        "happy": "开心",
        "brushing": "满足",
        "angry": "不满",
        "pain": "痛苦",
        "isolation": "焦虑"
    }
    return emotion_map.get(intent, "开心")
