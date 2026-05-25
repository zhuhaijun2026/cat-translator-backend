"""
语音识别服务（Speech-to-Text）
使用 Vosk 离线识别，不需要联网
"""
import os
import json
import wave
import subprocess
import urllib.request
import zipfile


class SpeechToText:
    """
    语音识别器 - Vosk离线版
    
    首次使用自动下载中文模型（约42MB），之后完全离线运行
    """

    MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"
    MODEL_NAME = "vosk-model-small-cn-0.22"
    MODEL_DIR = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'models',
        'vosk-model-small-cn'
    )

    def __init__(self):
        self.model = None
        self._ready = False
        self._try_load_model()

    def _try_load_model(self):
        """尝试加载已有模型"""
        try:
            from vosk import Model
            if os.path.exists(
                os.path.join(self.MODEL_DIR, 'conf', 'model.conf')
            ):
                self.model = Model(self.MODEL_DIR)
                self._ready = True
                print("✅ Vosk中文语音识别模型已加载")
            else:
                print(
                    "⚠️ Vosk中文模型未下载，"
                    "首次人语翻译时会自动下载（约42MB）"
                )
        except ImportError:
            print("⚠️ vosk未安装，请执行: pip install vosk")
        except Exception as e:
            print(f"⚠️ Vosk模型加载失败: {e}")

    def _download_model(self):
        """下载并解压中文模型"""
        os.makedirs(os.path.dirname(self.MODEL_DIR), exist_ok=True)
        zip_path = self.MODEL_DIR + '.zip'

        print("正在下载Vosk中文模型（约42MB），请稍候...")
        try:
            urllib.request.urlretrieve(self.MODEL_URL, zip_path)
        except Exception as e:
            raise ValueError(
                f"模型下载失败: {e}\n"
                "请手动下载: https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip\n"
                f"解压到: {os.path.dirname(self.MODEL_DIR)}"
            )

        print("下载完成，正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(os.path.dirname(self.MODEL_DIR))
        os.unlink(zip_path)

        extracted = os.path.join(
            os.path.dirname(self.MODEL_DIR), self.MODEL_NAME
        )
        if os.path.exists(extracted) and not os.path.exists(self.MODEL_DIR):
            os.rename(extracted, self.MODEL_DIR)

        print("✅ Vosk模型下载完成！")
        from vosk import Model
        self.model = Model(self.MODEL_DIR)
        self._ready = True

    def _convert_to_wav(self, audio_path: str) -> str:
        """将音频转为16kHz单声道wav"""
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == '.wav':
            # 确认格式正确
            try:
                wf = wave.open(audio_path, 'rb')
                if (wf.getframerate() == 16000 and
                        wf.getnchannels() == 1):
                    wf.close()
                    return audio_path
                wf.close()
            except Exception:
                pass

        wav_path = audio_path.rsplit('.', 1)[0] + '_stt.wav'
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', audio_path,
                 '-ar', '16000', '-ac', '1', wav_path],
                check=True, capture_output=True, timeout=10
            )
            return wav_path
        except (FileNotFoundError, subprocess.CalledProcessError):
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(wav_path, format='wav')
                return wav_path
            except Exception as e:
                raise ValueError(
                    f"音频格式转换失败(需要安装ffmpeg): {str(e)}"
                )

    def recognize(self, audio_path: str, language: str = "zh-CN") -> dict:
        """
        识别语音内容

        Args:
            audio_path: 音频文件路径
            language: 语言（Vosk用模型决定，此参数保留兼容）

        Returns:
            {"text": "识别的文字", "confidence": 0.9}
        """
        if not self._ready:
            self._download_model()

        wav_path = None
        try:
            wav_path = self._convert_to_wav(audio_path)

            from vosk import KaldiRecognizer
            wf = wave.open(wav_path, "rb")

            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)

            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    results.append(json.loads(rec.Result()))
            results.append(json.loads(rec.FinalResult()))

            wf.close()
            text = " ".join(
                r.get("text", "") for r in results
            ).strip()

            return {"text": text, "confidence": 0.9}

        except Exception as e:
            raise ValueError(f"语音识别失败: {str(e)}")
        finally:
            if wav_path and wav_path != audio_path:
                try:
                    os.unlink(wav_path)
                except:
                    pass
					