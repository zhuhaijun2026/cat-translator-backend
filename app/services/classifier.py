"""
猫叫分类器服务
基于 MFCC 特征提取 + 预训练模型
"""
import os
import subprocess
import numpy as np
import librosa
from typing import Dict, Optional


class CatSoundClassifier:
    """
    猫叫声分类器
    
    MVP阶段：使用基于规则的简单分类（根据音频特征）
    后续：替换为训练好的 CNN+LSTM 模型
    """
    
    # 分类标签
    LABELS = ["brushing", "food", "isolation", "happy", "angry", "pain"]
    
    # 各分类的MFCC特征参考值（简化版，后续用真实训练数据替换）
    REFERENCE_FEATURES = {
        "brushing": {"avg_mfcc_1": 5.0, "spectral_centroid": 1500, "rms": 0.02},
        "food": {"avg_mfcc_1": 8.0, "spectral_centroid": 2500, "rms": 0.05},
        "isolation": {"avg_mfcc_1": 10.0, "spectral_centroid": 3000, "rms": 0.04},
        "happy": {"avg_mfcc_1": 7.0, "spectral_centroid": 2200, "rms": 0.06},
        "angry": {"avg_mfcc_1": 12.0, "spectral_centroid": 3500, "rms": 0.08},
        "pain": {"avg_mfcc_1": 15.0, "spectral_centroid": 4000, "rms": 0.03}
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("⚠️ 未加载预训练模型，使用规则分类器（MVP模式）")
    
    def _load_model(self, model_path: str):
        """加载预训练模型（后续实现）"""
        pass

    def _convert_to_wav(self, audio_path: str) -> str:
        """将音频文件转换为wav格式（librosa不支持webm等格式）"""
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == '.wav':
            return audio_path
        
        wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', audio_path, '-ar', '16000', '-ac', '1', wav_path],
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
                raise ValueError(f"音频格式转换失败(需要安装ffmpeg): {str(e)}")
    
    def extract_features(self, audio_path: str) -> Dict:
        try:
            wav_path = self._convert_to_wav(audio_path)
            y, sr = librosa.load(wav_path, sr=16000)
            
            if len(y) < sr * 0.3:
                y = np.pad(y, (0, sr * 0.3 - len(y)))
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            duration = len(y) / sr
            
            return {
                "mfcc_mean": mfcc_mean.tolist(),
                "mfcc_std": mfcc_std.tolist(),
                "spectral_centroid": float(spectral_centroid),
                "rms": float(rms),
                "zcr": float(zcr),
                "spectral_bandwidth": float(spectral_bandwidth),
                "duration": duration
            }
            
        except Exception as e:
            raise ValueError(f"特征提取失败: {str(e)}")
    
    def predict(self, audio_path: str) -> Dict:
        features = self.extract_features(audio_path)
        
        if self.model:
            return self._predict_with_model(features)
        else:
            return self._predict_with_rules(features)
    
    def _predict_with_model(self, features: Dict) -> Dict:
        """使用预训练模型预测（后续实现）"""
        pass
    
    def _predict_with_rules(self, features: Dict) -> Dict:
        centroid = features["spectral_centroid"]
        rms = features["rms"]
        duration = features["duration"]
        zcr = features["zcr"]
        
        probs = {}
        
        if rms > 0.06 and centroid > 3000:
            probs = {
                "angry": 0.35,
                "food": 0.30,
                "pain": 0.15,
                "happy": 0.10,
                "isolation": 0.05,
                "brushing": 0.05
            }
        elif rms > 0.04 and centroid > 2000:
            probs = {
                "food": 0.35,
                "happy": 0.25,
                "isolation": 0.20,
                "angry": 0.10,
                "brushing": 0.05,
                "pain": 0.05
            }
        elif rms < 0.03 and centroid < 2000:
            probs = {
                "brushing": 0.45,
                "happy": 0.20,
                "pain": 0.15,
                "isolation": 0.10,
                "food": 0.05,
                "angry": 0.05
            }
        elif rms < 0.04 and centroid > 2500:
            probs = {
                "isolation": 0.35,
                "pain": 0.25,
                "food": 0.15,
                "brushing": 0.10,
                "happy": 0.10,
                "angry": 0.05
            }
        else:
            probs = {
                "food": 0.30,
                "happy": 0.25,
                "brushing": 0.15,
                "isolation": 0.15,
                "angry": 0.10,
                "pain": 0.05
            }
        
        intent = max(probs, key=probs.get)
        confidence = probs[intent]
        
        return {
            "intent": intent,
            "confidence": confidence,
            "all_probs": probs
        }

# 全局分类器实例
classifier = CatSoundClassifier()
