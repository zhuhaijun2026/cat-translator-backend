"""
语音识别服务（Speech-to-Text）
使用科大讯飞语音听写API（免费，每天500次）
"""
import os
import json
import hmac
import hashlib
import base64
import subprocess
import threading
from datetime import datetime, timezone
from urllib.parse import quote

try:
    import websocket  # websocket-client包
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

try:
    import httpx
    _HTTP = "httpx"
except ImportError:
    _HTTP = None


class SpeechToText:
    """
    语音识别器
    使用科大讯飞语音听写WebSocket API
    免费额度：每天500次
    """
    
    def __init__(self):
        self.app_id = os.environ.get("XFYUN_APP_ID", "")
        self.api_key = os.environ.get("XFYUN_API_KEY", "")
        self.api_secret = os.environ.get("XFYUN_API_SECRET", "")
        
        if self.app_id and self.api_key and self.api_secret and _WS_AVAILABLE:
            print("✅ 讯飞语音听写已配置（每天免费500次）")
        else:
            missing = []
            if not self.app_id: missing.append("XFYUN_APP_ID")
            if not self.api_key: missing.append("XFYUN_API_KEY")
            if not self.api_secret: missing.append("XFYUN_API_SECRET")
            if not _WS_AVAILABLE: missing.append("websocket-client包")
            print(f"⚠️ 语音识别未配置，缺少: {', '.join(missing)}")
            print("⚠️ 请前往 https://www.xfyun.cn 注册并创建语音听写应用")
    
    def _create_auth_url(self):
        """生成讯飞WebSocket鉴权URL"""
        url = "wss://iat-api.xfyun.cn/v2/iat"
        now = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
        
        signature_origin = f"host: iat-api.xfyun.cn\ndate: {now}\nGET /v2/iat HTTP/1.1"
        signature_sha = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            hashlib.sha256
        ).digest()
        signature = base64.b64encode(signature_sha).decode('utf-8')
        
        authorization_origin = (
            f'api_key="{self.api_key}", '
            f'algorithm="hmac-sha256", '
            f'headers="host date request-line", '
            f'signature="{signature}"'
        )
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        
        return f"{url}?authorization={quote(authorization)}&date={quote(now)}&host=iat-api.xfyun.cn"
    
    def _convert_to_pcm(self, audio_path: str) -> str:
        """将音频转换为16kHz 16bit 单声道PCM（讯飞要求的格式）"""
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == '.pcm':
            return audio_path
        
        pcm_path = audio_path.rsplit('.', 1)[0] + '_stt.pcm'
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', audio_path,
             '-f', 's16le', '-ar', '16000', '-ac', '1', pcm_path],
            capture_output=True, timeout=15
        )
        if result.returncode != 0 or not os.path.exists(pcm_path):
            raise ValueError("音频格式转换失败")
        return pcm_path
    
    def recognize(self, audio_path: str, language: str = "zh-CN") -> dict:
        """
        识别语音内容
        使用讯飞语音听写WebSocket API
        
        Returns:
            {"text": "识别的文字", "confidence": 0.9}
        """
        if not (self.app_id and self.api_key and self.api_secret and _WS_AVAILABLE):
            print("⚠️ 讯飞语音识别未配置")
            return {"text": "", "confidence": 0.0}
        
        pcm_path = None
        try:
            pcm_path = self._convert_to_pcm(audio_path)
            
            with open(pcm_path, 'rb') as f:
                audio_data = f.read()
            
            if len(audio_data) == 0:
                print("⚠️ 音频数据为空")
                return {"text": "", "confidence": 0.0}
            
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            ws_url = self._create_auth_url()
            
            result_text = []
            result_received = threading.Event()
            error_msg = [None]
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    code = data.get("code", -1)
                    if code != 0:
                        error_msg[0] = f"讯飞错误({code}): {data.get('message', '')}"
                        result_received.set()
                        return
                    
                    data_section = data.get("data", {})
                    result_section = data_section.get("result", {})
                    ws_list = result_section.get("ws", [])
                    
                    for ws_item in ws_list:
                        for cw in ws_item.get("cw", []):
                            result_text.append(cw.get("w", ""))
                    
                    if data_section.get("status") == 2:
                        result_received.set()
                except Exception as e:
                    error_msg[0] = str(e)
                    result_received.set()
            
            def on_error(ws, error):
                error_msg[0] = str(error)
                result_received.set()
            
            def on_close(ws, close_status, close_msg):
                result_received.set()
            
            def on_open(ws):
                request = {
                    "common": {"app_id": self.app_id},
                    "business": {
                        "language": "zh_cn",
                        "domain": "iat",
                        "accent": "mandarin",
                        "vad_eos": 2000,
                        "dwa": "wpgs"  # 动态修正
                    },
                    "data": {
                        "status": 2,  # 2=最后一帧（一次性发送）
                        "format": "audio/L16;rate=16000",
                        "encoding": "raw",
                        "audio": audio_base64
                    }
                }
                ws.send(json.dumps(request))
            
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            wst = threading.Thread(target=ws.run_forever)
            wst.daemon = True
            wst.start()
            
            result_received.wait(timeout=15)
            
            try:
                ws.close()
            except:
                pass
            
            if error_msg[0]:
                print(f"⚠️ 讯飞识别失败: {error_msg[0]}")
                return {"text": "", "confidence": 0.0}
            
            text = "".join(result_text).strip()
            if text:
                print(f"✅ 讯飞识别成功: '{text}'")
                return {"text": text, "confidence": 0.9}
            else:
                print("⚠️ 讯飞识别结果为空")
                return {"text": "", "confidence": 0.0}
                
        except ValueError as e:
            print(f"⚠️ 音频处理失败: {e}")
            return {"text": "", "confidence": 0.0}
        except Exception as e:
            print(f"⚠️ 语音识别异常: {e}")
            return {"text": "", "confidence": 0.0}
        finally:
            if pcm_path and pcm_path != audio_path:
                try:
                    os.unlink(pcm_path)
                except:
                    pass
