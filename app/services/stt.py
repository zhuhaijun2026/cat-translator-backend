"""
语音识别服务（Speech-to-Text）
使用 Vosk 离线识别，无需联网
"""
import os
import json
import wave
import subprocess


class SpeechToText:
    """
    Vosk 离线语音识别器
    """
    
    def __init__(self):
        self.model = None
        self.model_available = False
        self._load_model()
    
    def _load_model(self):
        """加载Vosk中文模型"""
        try:
            import vosk
            # 模型路径：优先环境变量，其次默认路径
            model_path = os.environ.get(
                "VOSK_MODEL_PATH",
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "vosk-model-small-cn")
            )
            if not os.path.exists(model_path):
                print(f"⚠️ Vosk模型目录不存在: {model_path}")
                print("尝试自动下载模型...")
                self._download_model(model_path)
            
            if os.path.exists(os.path.join(model_path, "conf", "model.conf")):
                self.model = vosk.Model(model_path)
                self.model_available = True
                print(f"✅ Vosk中文模型加载成功: {model_path}")
            else:
                print(f"⚠️ Vosk模型文件不完整: {model_path}")
                print("⚠️ 语音识别将不可用")
        except ImportError:
            print("⚠️ vosk包未安装，语音识别不可用")
        except Exception as e:
            print(f"⚠️ Vosk模型加载失败: {e}")
            print("⚠️ 语音识别将不可用")
    
    def _download_model(self, model_path):
        """自动下载Vosk中文模型"""
        import zipfile
        import urllib.request
        
        url = "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"
        zip_path = model_path + ".zip"
        
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"正在下载Vosk中文模型（约42MB）...")
            urllib.request.urlretrieve(url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(os.path.dirname(model_path))
            
            # 重命名解压后的目录
            extracted = os.path.join(os.path.dirname(model_path), "vosk-model-small-cn-0.22")
            if os.path.exists(extracted) and not os.path.exists(model_path):
                os.rename(extracted, model_path)
            
            os.remove(zip_path)
            print("✅ Vosk模型下载完成")
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
    
    def _convert_to_wav(self, audio_path: str) -> str:
        """将音频文件转换为16kHz单声道WAV（Vosk要求的格式）"""
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == '.wav':
            # 检查是否已经是16kHz单声道
            try:
                with wave.open(audio_path, 'rb') as wf:
                    if wf.getframerate() == 16000 and wf.getnchannels() == 1:
                        return audio_path
            except:
                pass
        
        wav_path = audio_path.rsplit('.', 1)[0] + '_stt.wav'
        try:
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', audio_path,
                 '-ar', '16000', '-ac', '1', wav_path],
                capture_output=True, timeout=10
            )
            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8', errors='replace')
                print(f"⚠️ ffmpeg转换失败: {stderr[:200]}")
                raise ValueError("音频格式转换失败")
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                raise ValueError("音频格式转换失败：输出文件为空")
            return wav_path
        except FileNotFoundError:
            raise ValueError("音频格式转换失败(需要安装ffmpeg)")
        except subprocess.TimeoutExpired:
            raise ValueError("音频格式转换超时")
    
    def recognize(self, audio_path: str, language: str = "zh-CN") -> dict:
        """
        识别语音内容
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码（Vosk中文模型只支持中文）
            
        Returns:
            {"text": "识别的文字", "confidence": 0.9}
        """
        # 模型不可用时直接返回空结果，不抛异常
        if not self.model_available or not self.model:
            print("⚠️ Vosk模型不可用，无法进行语音识别")
            return {"text": "", "confidence": 0.0}
        
        import vosk
        wav_path = None
        try:
            wav_path = self._convert_to_wav(audio_path)
            
            wf = wave.open(wav_path, "rb")
            rec = vosk.KaldiRecognizer(self.model, wf.getframerate())
            
            text_parts = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        text_parts.append(result["text"])
            
            # 获取最终结果
            final_result = json.loads(rec.FinalResult())
            if final_result.get("text"):
                text_parts.append(final_result["text"])
            
            wf.close()
            recognized_text = " ".join(text_parts).strip()
            confidence = 0.9 if recognized_text else 0.0
            
            print(f"✅ 语音识别结果: '{recognized_text}' (置信度: {confidence})")
            return {"text": recognized_text, "confidence": confidence}
            
        except ValueError as e:
            # 音频格式转换失败
            print(f"⚠️ 音频处理失败: {e}")
            return {"text": "", "confidence": 0.0}
        except Exception as e:
            print(f"⚠️ 语音识别异常: {e}")
            return {"text": "", "confidence": 0.0}
        finally:
            if wav_path and wav_path != audio_path:
                try:
                    os.unlink(wav_path)
                except:
                    pass
