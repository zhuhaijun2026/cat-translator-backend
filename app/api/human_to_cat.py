"""
人语转猫叫 API
支持两种模式：
1. 文字输入 → 猫叫映射（/human-to-cat）
2. 音频输入 → STT转文字 → 猫叫映射（/human-audio-to-cat）
"""
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.services.stt import SpeechToText

router = APIRouter()

stt = SpeechToText()

# 人语意图 → 猫叫映射（按关键词长度降序，长词优先匹配）
HUMAN_TO_CAT_MAP = [
    # brushing（满足/摸摸/梳毛） → purr
    {"keyword": "梳毛", "intent": "brushing", "cat_sound": "purr",
     "reply": "🐱 呼噜噜～梳毛好舒服～"},
    {"keyword": "摸摸", "intent": "brushing", "cat_sound": "purr",
     "reply": "🐱 呼噜噜～摸摸～"},
    {"keyword": "摸摸你", "intent": "brushing", "cat_sound": "purr",
     "reply": "🐱 呼噜噜～再摸摸～"},
    {"keyword": "摸摸头", "intent": "brushing", "cat_sound": "purr",
     "reply": "🐱 呼噜噜～头也摸摸～"},
    {"keyword": "揉揉", "intent": "brushing", "cat_sound": "purr",
     "reply": "🐱 呼噜噜～好舒服～"},
    {"keyword": "挠挠", "intent": "brushing", "cat_sound": "purr",
     "reply": "🐱 呼噜噜～挠挠下巴～"},
    {"keyword": "乖宝宝", "intent": "brushing", "cat_sound": "purr",
     "reply": "🐱 呼噜噜～最乖了～"},
    {"keyword": "晚安", "intent": "brushing", "cat_sound": "purr",
     "reply": "🐱 呼噜噜～晚安～"},
    {"keyword": "睡觉", "intent": "brushing", "cat_sound": "purr",
     "reply": "🐱 呼噜噜～晚安～"},
    {"keyword": "摸", "intent": "brushing", "cat_sound": "purr",
     "reply": "🐱 呼噜噜～摸摸～"},
    {"keyword": "撸", "intent": "brushing", "cat_sound": "purr",
     "reply": "🐱 呼噜噜～撸猫～"},

    # food（饿了/吃饭） → meow_hungry
    {"keyword": "肚子饿了", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 喵！开饭啦！"},
    {"keyword": "饿了吗", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 翻译中：你饿不饿？"},
    {"keyword": "吃饭了", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 喵～开饭啦！"},
    {"keyword": "想吃", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 喵～想吃东西！"},
    {"keyword": "给吃的", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 喵～给我吃的！"},
    {"keyword": "猫粮", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 喵～猫粮！"},
    {"keyword": "罐头", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 喵～罐头！"},
    {"keyword": "零食", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 喵～零食！"},
    {"keyword": "饿了", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 喵～好饿！"},
    {"keyword": "喂你", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 喵～开饭啦！"},
    {"keyword": "喂猫", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 喵～开饭啦！"},
    {"keyword": "饿", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 收到！我帮你告诉猫咪～"},
    {"keyword": "吃", "intent": "food", "cat_sound": "meow_hungry",
     "reply": "🐱 喵～开饭啦！"},

    # angry（不满/制止） → hiss
    {"keyword": "不要这样", "intent": "angry", "cat_sound": "hiss",
     "reply": "🐱 嘶～不可以这样！"},
    {"keyword": "不许", "intent": "angry", "cat_sound": "hiss",
     "reply": "🐱 嘶～不许！"},
    {"keyword": "别闹", "intent": "angry", "cat_sound": "hiss",
     "reply": "🐱 嘶～别闹了！"},
    {"keyword": "下去", "intent": "angry", "cat_sound": "hiss",
     "reply": "🐱 嘶～下去！"},
    {"keyword": "不行", "intent": "angry", "cat_sound": "hiss",
     "reply": "🐱 嘶～不可以！"},
    {"keyword": "不要", "intent": "angry", "cat_sound": "hiss",
     "reply": "🐱 嘶～不准这样！"},
    {"keyword": "停", "intent": "angry", "cat_sound": "hiss",
     "reply": "🐱 嘶～停下！"},
    {"keyword": "洗澡", "intent": "angry", "cat_sound": "hiss",
     "reply": "🐱 嘶～不要洗澡！"},
    {"keyword": "坏猫", "intent": "angry", "cat_sound": "hiss",
     "reply": "🐱 嘶～我才不坏！"},

    # pain（痛苦/受伤） → meow_pain
    {"keyword": "不舒服", "intent": "pain", "cat_sound": "meow_pain",
     "reply": "🐱 喵呜...哪里不舒服？"},
    {"keyword": "受伤", "intent": "pain", "cat_sound": "meow_pain",
     "reply": "🐱 喵呜...好疼..."},
    {"keyword": "生病", "intent": "pain", "cat_sound": "meow_pain",
     "reply": "🐱 喵呜...我不舒服..."},
    {"keyword": "痛", "intent": "pain", "cat_sound": "meow_pain",
     "reply": "🐱 喵呜...好痛..."},
    {"keyword": "疼", "intent": "pain", "cat_sound": "meow_pain",
     "reply": "🐱 喵呜...好疼..."},

    # isolation（焦虑/害怕） → meow_anxious
    {"keyword": "一个人", "intent": "isolation", "cat_sound": "meow_anxious",
     "reply": "🐱 喵？不要留我一个人..."},
    {"keyword": "害怕", "intent": "isolation", "cat_sound": "meow_anxious",
     "reply": "🐱 喵？别怕，我在～"},
    {"keyword": "出去", "intent": "isolation", "cat_sound": "meow_anxious",
     "reply": "🐱 喵？带我出去吗？"},
    {"keyword": "别走", "intent": "isolation", "cat_sound": "meow_anxious",
     "reply": "🐱 喵？别离开我..."},
    {"keyword": "孤独", "intent": "isolation", "cat_sound": "meow_anxious",
     "reply": "🐱 喵？好孤单..."},

    # happy（开心/友好） → meow_happy
    {"keyword": "过来呀", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～来啦！"},
    {"keyword": "我爱你", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～我也爱你！"},
    {"keyword": "喜欢你", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～我也喜欢你！"},
    {"keyword": "好孩子", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～我是好孩子！"},
    {"keyword": "漂亮", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～当然漂亮！"},
    {"keyword": "可爱", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～人家很可爱的～"},
    {"keyword": "宝贝", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～叫我吗？"},
    {"keyword": "过来", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～来啦！"},
    {"keyword": "早安", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～早上好！"},
    {"keyword": "你好", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～你好呀！"},
    {"keyword": "爱你", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～我也爱你！"},
    {"keyword": "乖", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵呜～真乖！"},
    {"keyword": "玩", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵！一起玩！"},
    {"keyword": "回家", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵～回家啦！"},
    {"keyword": "无聊", "intent": "happy", "cat_sound": "meow_happy",
     "reply": "🐱 喵！陪我玩！"},
]

DEFAULT_RESPONSE = {
    "intent": "happy",
    "cat_sound": "meow_happy",
    "reply": "🐱 喵～让我想想怎么翻译..."
}


def match_text_to_cat(text: str) -> dict:
    """按关键词长度降序匹配，长词优先"""
    text = text.strip().lower()
    sorted_map = sorted(HUMAN_TO_CAT_MAP, key=lambda x: len(x["keyword"]), reverse=True)
    for item in sorted_map:
        if item["keyword"] in text:
            return item
    return DEFAULT_RESPONSE


class HumanToCatRequest(BaseModel):
    text: str


@router.post("/human-to-cat")
async def human_to_cat(req: HumanToCatRequest):
    """人语文字 → 猫叫"""
    text = req.text.strip().lower()
    result = match_text_to_cat(text)
    return {
        "intent": result["intent"],
        "catSoundUrl": f"/static/audio/cats/{result['cat_sound']}.mp3",
        "reply": result["reply"]
    }


@router.post("/human-audio-to-cat")
async def human_audio_to_cat(audio: UploadFile = File(...)):
    """
    人语音频 → STT转文字 → 猫叫映射
    流程：上传音频 → 百度云语音识别 → 关键词匹配 → 返回猫叫
    """
    if not audio:
        raise HTTPException(status_code=400, detail="请上传音频文件")
    
    tmp_path = None
    try:
        filename = audio.filename or "audio.mp3"
        suffix = os.path.splitext(filename)[1]
        if not suffix:
            suffix = '.mp3'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"[human-audio-to-cat] 收到音频: {filename}, 大小: {len(content)} bytes")
        
        # 语音识别（stt内部优先百度云，备用Vosk）
        stt_result = stt.recognize(tmp_path, language="zh-CN")
        recognized_text = stt_result.get("text", "")
        stt_confidence = stt_result.get("confidence", 0.0)
        
        print(f"[human-audio-to-cat] 识别结果: '{recognized_text}' (置信度: {stt_confidence})")
        
        if recognized_text:
            cat_result = match_text_to_cat(recognized_text)
        else:
            cat_result = DEFAULT_RESPONSE
            recognized_text = "(未识别到语音内容)"
        
        return {
            "text": recognized_text,
            "emotion": _intent_to_emotion(cat_result["intent"]),
            "confidence": int(stt_confidence * 100),
            "catSoundUrl": f"/static/audio/cats/{cat_result['cat_sound']}.mp3",
            "reply": cat_result["reply"],
            "intent": cat_result["intent"]
        }
        
    except Exception as e:
        print(f"[human-audio-to-cat] ❌ 处理失败: {e}")
        return {
            "text": "(翻译服务暂时不可用)",
            "emotion": "困惑",
            "confidence": 0,
            "catSoundUrl": "/static/audio/cats/meow_happy.mp3",
            "reply": "🐱 喵？翻译服务暂时不可用，请稍后再试",
            "intent": "happy"
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except:
                pass


def _intent_to_emotion(intent: str) -> str:
    emotion_map = {
        "food": "急切",
        "happy": "开心",
        "brushing": "满足",
        "angry": "不满",
        "pain": "痛苦",
        "isolation": "焦虑"
    }
    return emotion_map.get(intent, "开心")
