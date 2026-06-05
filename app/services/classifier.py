"""
猫叫分类器服务 v2
- 使用多特征评分系统替代简单if-else
- 支持加载sklearn预训练模型（优先）
- 规则分类器基于猫叫行为学研究校准
- 文件名保持 classifier.py，直接替换 app/services/classifier.py
"""
import os
import subprocess
import json
import numpy as np
import librosa
from typing import Dict, Optional


class CatSoundClassifier:
    """
    猫叫声分类器 v2
    
    改进：
    1. 使用8+维特征评分，而非2维if-else
    2. 基于猫叫行为学研究校准阈值
    3. 支持sklearn模型（RandomForest等）
    """
    
    LABELS = ["brushing", "food", "isolation", "happy", "angry", "pain"]
    
    # 类别中文映射
    LABEL_CN = {
        "brushing": "满足(呼噜)",
        "food": "急切(讨食)",
        "isolation": "焦虑(呼唤)",
        "happy": "开心(互动)",
        "angry": "不满(威吓)",
        "pain": "痛苦(哀鸣)"
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.feature_profiles = None
        
        # 尝试加载预训练模型
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        
        # 尝试加载特征参考profile
        profile_path = os.path.join(os.path.dirname(model_path or ""), "feature_profiles.json")
        if os.path.exists(profile_path):
            with open(profile_path, "r") as f:
                self.feature_profiles = json.load(f)
            print("✅ 已加载特征参考profile")
        else:
            print("⚠️ 未找到特征profile，使用评分规则分类器")
    
    def _load_model(self, model_path: str):
        """加载sklearn预训练模型"""
        try:
            import joblib
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                self.model = model_data.get("model")
                self.scaler = model_data.get("scaler")
            else:
                self.model = model_data
            print(f"✅ 已加载预训练模型: {model_path}")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}，降级为规则分类器")
            self.model = None

    def _convert_to_wav(self, audio_path: str) -> str:
        """将音频文件转换为wav格式"""
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
        """提取完整音频特征（8维+）"""
        try:
            wav_path = self._convert_to_wav(audio_path)
            y, sr = librosa.load(wav_path, sr=16000)
            
            # 短音频补零
            if len(y) < sr * 0.3:
                y = np.pad(y, (0, int(sr * 0.3) - len(y)))
            
            # === 基础MFCC特征 ===
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # === 频谱特征 ===
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # === 能量与时域特征 ===
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            duration = len(y) / sr
            
            # === 频谱对比度（区分谐波vs噪声） ===
            try:
                spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
            except:
                spectral_contrast = np.zeros(7)
            
            # === 频谱平坦度（噪声性指标，越高越像噪声） ===
            try:
                spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            except:
                spectral_flatness = 0.0
            
            # === 音高相关特征 ===
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                    sr=sr
                )
                f0_valid = f0[~np.isnan(f0)]
                if len(f0_valid) > 0:
                    f0_mean = float(np.mean(f0_valid))
                    f0_std = float(np.std(f0_valid))
                    f0_range = float(np.max(f0_valid) - np.min(f0_valid))
                    voiced_ratio = float(np.sum(voiced_flag) / len(voiced_flag))
                else:
                    f0_mean = 0.0
                    f0_std = 0.0
                    f0_range = 0.0
                    voiced_ratio = 0.0
            except:
                f0_mean = 0.0
                f0_std = 0.0
                f0_range = 0.0
                voiced_ratio = 0.0
            
            # === RMS变化率（节奏性指标） ===
            rms_frame = librosa.feature.rms(y=y)[0]
            if len(rms_frame) > 1:
                rms_std = float(np.std(rms_frame))
                # 检测周期性（呼噜的节奏）
                rms_diff = np.diff(rms_frame)
                rms_periodicity = float(1.0 - np.mean(np.abs(rms_diff)) / (np.mean(rms_frame) + 1e-8))
            else:
                rms_std = 0.0
                rms_periodicity = 0.0
            
            return {
                "mfcc_mean": mfcc_mean.tolist(),
                "mfcc_std": mfcc_std.tolist(),
                "spectral_centroid": float(spectral_centroid),
                "spectral_bandwidth": float(spectral_bandwidth),
                "spectral_rolloff": float(spectral_rolloff),
                "rms": float(rms),
                "zcr": float(zcr),
                "duration": duration,
                "spectral_contrast": spectral_contrast.tolist() if isinstance(spectral_contrast, np.ndarray) else spectral_contrast,
                "spectral_flatness": float(spectral_flatness),
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "f0_range": f0_range,
                "voiced_ratio": voiced_ratio,
                "rms_std": rms_std,
                "rms_periodicity": rms_periodicity
            }
            
        except Exception as e:
            raise ValueError(f"特征提取失败: {str(e)}")
    
    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """将特征字典转为模型输入向量"""
        vec = []
        # MFCC mean (13) + std (13)
        vec.extend(features.get("mfcc_mean", [0]*13))
        vec.extend(features.get("mfcc_std", [0]*13))
        # 频谱特征 (5)
        vec.append(features.get("spectral_centroid", 0))
        vec.append(features.get("spectral_bandwidth", 0))
        vec.append(features.get("spectral_rolloff", 0))
        vec.append(features.get("spectral_flatness", 0))
        vec.append(features.get("zcr", 0))
        # 能量与时域 (4)
        vec.append(features.get("rms", 0))
        vec.append(features.get("rms_std", 0))
        vec.append(features.get("rms_periodicity", 0))
        vec.append(features.get("duration", 0))
        # 音高特征 (4)
        vec.append(features.get("f0_mean", 0))
        vec.append(features.get("f0_std", 0))
        vec.append(features.get("f0_range", 0))
        vec.append(features.get("voiced_ratio", 0))
        # 频谱对比度 (7)
        vec.extend(features.get("spectral_contrast", [0]*7))
        return np.array(vec, dtype=np.float32).reshape(1, -1)
    
    def predict(self, audio_path: str) -> Dict:
        features = self.extract_features(audio_path)
        
        if self.model is not None:
            return self._predict_with_model(features)
        else:
            return self._predict_with_rules(features)
    
    def _predict_with_model(self, features: Dict) -> Dict:
        """使用预训练sklearn模型预测"""
        try:
            X = self._features_to_vector(features)
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            probs_array = self.model.predict_proba(X)[0]
            probs = {k: float(v) for k, v in zip(self.LABELS, probs_array)}
            intent = max(probs, key=probs.get)
            confidence = probs[intent]
            
            return {
                "intent": intent,
                "confidence": confidence,
                "all_probs": probs
            }
        except Exception as e:
            print(f"模型预测失败: {e}，降级为规则分类器")
            return self._predict_with_rules(features)
    
    def _predict_with_rules(self, features: Dict) -> Dict:
        """
        多特征评分规则分类器 v2
        
        基于猫叫行为学研究：
        - 呼噜(purring): 低频(25-150Hz谐波), 低ZCR, 有节奏, 低centroid
        - 讨食meow: 中频, 短-中等时长, 中等能量
        - 焦虑meow: 高频, 较长, 音高变化大
        - 开心chirp: 短, 中频, 上扬音高
        - 威吓hiss: 极高ZCR(噪声), 高flatness, 宽频带
        - 痛苦howl: 长时, 高音高变化, 高能量
        """
        centroid = features.get("spectral_centroid", 0)
        rms = features.get("rms", 0)
        duration = features.get("duration", 0)
        zcr = features.get("zcr", 0)
        bandwidth = features.get("spectral_bandwidth", 0)
        flatness = features.get("spectral_flatness", 0)
        f0_mean = features.get("f0_mean", 0)
        f0_std = features.get("f0_std", 0)
        f0_range = features.get("f0_range", 0)
        voiced_ratio = features.get("voiced_ratio", 0)
        rms_periodicity = features.get("rms_periodicity", 0)
        mfcc_mean = features.get("mfcc_mean", [0]*13)
        mfcc_1 = mfcc_mean[0] if len(mfcc_mean) > 0 else 0
        mfcc_2 = mfcc_mean[1] if len(mfcc_mean) > 1 else 0
        
        scores = {label: 0.0 for label in self.LABELS}
        
        # ============================================================
        # BRUSHING (呼噜/满足)
        # 核心特征: 极低频、低ZCR、有节奏性、低spectral_flatness
        # ============================================================
        if centroid < 1000:
            scores["brushing"] += 5  # 极低centroid是呼噜的强信号
        elif centroid < 1500:
            scores["brushing"] += 2.5
        elif centroid < 2000:
            scores["brushing"] += 0.5
        
        if zcr < 0.03:
            scores["brushing"] += 4  # 呼噜是谐波信号，ZCR极低
        elif zcr < 0.06:
            scores["brushing"] += 2
        elif zcr < 0.10:
            scores["brushing"] += 0.5
        
        if flatness < 0.05:
            scores["brushing"] += 3  # 高度谐波性
        elif flatness < 0.1:
            scores["brushing"] += 1.5
        
        if rms_periodicity > 0.7:
            scores["brushing"] += 3  # 呼噜有节奏
        elif rms_periodicity > 0.5:
            scores["brushing"] += 1.5
        
        if bandwidth < 1200:
            scores["brushing"] += 2
        elif bandwidth < 1800:
            scores["brushing"] += 1
        
        if f0_mean > 0 and f0_mean < 200:
            scores["brushing"] += 2  # 极低基频
        elif f0_mean > 0 and f0_mean < 400:
            scores["brushing"] += 0.5
        
        # ============================================================
        # FOOD (讨食meow)
        # 核心特征: 中频、中等能量、短-中等时长、谐波性适中
        # ============================================================
        if 1500 < centroid < 3000:
            scores["food"] += 2.5
        elif 1200 < centroid < 3500:
            scores["food"] += 1
        
        if 0.03 < rms < 0.08:
            scores["food"] += 2
        elif 0.02 < rms < 0.10:
            scores["food"] += 1
        
        if 0.3 < duration < 1.5:
            scores["food"] += 2
        elif duration < 2.5:
            scores["food"] += 0.5
        
        if 0.04 < zcr < 0.10:
            scores["food"] += 1.5
        elif zcr < 0.15:
            scores["food"] += 0.5
        
        if 0.02 < flatness < 0.15:
            scores["food"] += 1  # 中等谐波性
        
        if 1500 < bandwidth < 3000:
            scores["food"] += 1
        
        # ============================================================
        # ISOLATION (焦虑/害怕的呼唤)
        # 核心特征: 高频、较长时长、音高变化大、中高能量
        # ============================================================
        if centroid > 2800:
            scores["isolation"] += 3.5
        elif centroid > 2200:
            scores["isolation"] += 2
        elif centroid > 1800:
            scores["isolation"] += 0.5
        
        if duration > 1.2:
            scores["isolation"] += 3  # 焦虑叫声通常较长
        elif duration > 0.7:
            scores["isolation"] += 1.5
        
        if f0_std > 80:
            scores["isolation"] += 2.5  # 音高波动大
        elif f0_std > 40:
            scores["isolation"] += 1
        
        if f0_range > 300:
            scores["isolation"] += 2  # 音高范围大
        elif f0_range > 150:
            scores["isolation"] += 1
        
        if rms > 0.04:
            scores["isolation"] += 1
        elif rms > 0.02:
            scores["isolation"] += 0.5
        
        if zcr > 0.08:
            scores["isolation"] += 1
        
        if bandwidth > 2500:
            scores["isolation"] += 1.5
        elif bandwidth > 2000:
            scores["isolation"] += 0.5
        
        # ============================================================
        # HAPPY (开心chirp/trill)
        # 核心特征: 短时、中频、上扬音高、中等能量
        # ============================================================
        if duration < 0.5:
            scores["happy"] += 3.5  # chirp非常短
        elif duration < 0.8:
            scores["happy"] += 2
        elif duration < 1.2:
            scores["happy"] += 0.5
        
        if 1800 < centroid < 2800:
            scores["happy"] += 1.5
        elif 1500 < centroid < 3200:
            scores["happy"] += 0.5
        
        if 0.02 < rms < 0.06:
            scores["happy"] += 1
        
        if 0.04 < zcr < 0.10:
            scores["happy"] += 1
        
        if f0_range > 100 and duration < 0.8:
            scores["happy"] += 1.5  # 短时+音高变化=chirp特征
        
        # ============================================================
        # ANGRY (威吓hiss/growl)
        # 核心特征: 极高ZCR(噪声)、高flatness、宽频带
        # hiss: 纯噪声, growl: 低频+噪声
        # ============================================================
        if zcr > 0.25:
            scores["angry"] += 5  # hiss的ZCR极高，这是最强信号
        elif zcr > 0.18:
            scores["angry"] += 3.5
        elif zcr > 0.12:
            scores["angry"] += 1.5
        
        if flatness > 0.3:
            scores["angry"] += 4  # 高噪声性
        elif flatness > 0.15:
            scores["angry"] += 2
        elif flatness > 0.08:
            scores["angry"] += 0.5
        
        if centroid > 3500:
            scores["angry"] += 2.5
        elif centroid > 2500:
            scores["angry"] += 1
        
        if bandwidth > 3500:
            scores["angry"] += 2
        elif bandwidth > 2500:
            scores["angry"] += 1
        
        if rms > 0.06:
            scores["angry"] += 1.5
        elif rms > 0.04:
            scores["angry"] += 0.5
        
        # growl特有: 低频但高flatness(噪声叠加低频)
        if centroid < 2000 and flatness > 0.15:
            scores["angry"] += 2
        
        # ============================================================
        # PAIN (痛苦howl/yowl)
        # 核心特征: 长时、高音高变化、高能量、谐波+噪声混合
        # ============================================================
        if duration > 2.0:
            scores["pain"] += 5  # 哀鸣很长
        elif duration > 1.5:
            scores["pain"] += 3
        elif duration > 1.0:
            scores["pain"] += 1
        
        if f0_std > 100:
            scores["pain"] += 3  # 剧烈音高波动
        elif f0_std > 60:
            scores["pain"] += 1.5
        
        if f0_range > 400:
            scores["pain"] += 2
        elif f0_range > 200:
            scores["pain"] += 1
        
        if rms > 0.05:
            scores["pain"] += 1.5
        elif rms > 0.03:
            scores["pain"] += 0.5
        
        if centroid > 2000:
            scores["pain"] += 1
        
        if 0.05 < flatness < 0.2:
            scores["pain"] += 1  # 哀鸣有一定谐波性但混合噪声
        
        # ============================================================
        # 软最大化 (Softmax) 转为概率
        # ============================================================
        score_array = np.array([scores[k] for k in self.LABELS])
        
        # 如果最高分太低(没有强信号), 给默认分布一个分量
        if np.max(score_array) < 3:
            score_array += 1.0  # 加基础分避免极端分布
        
        # Softmax with temperature
        temperature = 1.5
        score_shifted = score_array - np.max(score_array)
        exp_scores = np.exp(score_shifted / temperature)
        probs_array = exp_scores / exp_scores.sum()
        
        probs = {k: float(v) for k, v in zip(self.LABELS, probs_array)}
        intent = max(probs, key=probs.get)
        confidence = probs[intent]
        
        return {
            "intent": intent,
            "confidence": confidence,
            "all_probs": probs
        }


# 全局分类器实例
classifier = CatSoundClassifier()
