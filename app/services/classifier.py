"""
猫叫分类器服务 v3
- 修复手机录音特征偏差问题（降噪/AGC导致zcr和flatness偏低）
- 移除不可靠的rms_periodicity指标
- 以centroid为主判据（受手机处理影响最小）
- 新增调试接口 /api/v1/extract-features
"""
import os
import subprocess
import json
import numpy as np
import librosa
from typing import Dict, Optional


class CatSoundClassifier:
    """
    猫叫声分类器 v3
    
    v3 关键修复:
    1. 移除rms_periodicity（几乎所有录音都是0.9+，无区分度）
    2. 收紧brushing判定：必须flatness极低(<0.01)且centroid极低(<1200)
    3. 以centroid为最主要判据（受手机AGC/降噪影响最小）
    4. 增加频谱rolloff和对比度作为辅助判据
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
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("⚠️ 未加载预训练模型，使用v3评分规则分类器")
    
    def _load_model(self, model_path: str):
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
        """提取完整音频特征"""
        try:
            wav_path = self._convert_to_wav(audio_path)
            y, sr = librosa.load(wav_path, sr=16000)
            
            if len(y) < sr * 0.3:
                y = np.pad(y, (0, int(sr * 0.3) - len(y)))
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # 频谱特征
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # 能量与时域
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            duration = len(y) / sr
            
            # 频谱对比度
            try:
                spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
            except:
                spectral_contrast = np.zeros(7)
            
            # 频谱平坦度
            try:
                spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            except:
                spectral_flatness = 0.0
            
            # 音高
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
                )
                f0_valid = f0[~np.isnan(f0)]
                if len(f0_valid) > 0:
                    f0_mean = float(np.mean(f0_valid))
                    f0_std = float(np.std(f0_valid))
                    f0_range = float(np.max(f0_valid) - np.min(f0_valid))
                    voiced_ratio = float(np.sum(voiced_flag) / len(voiced_flag))
                else:
                    f0_mean, f0_std, f0_range, voiced_ratio = 0, 0, 0, 0
            except:
                f0_mean, f0_std, f0_range, voiced_ratio = 0, 0, 0, 0
            
            # RMS帧级统计
            rms_frame = librosa.feature.rms(y=y)[0]
            rms_std = float(np.std(rms_frame)) if len(rms_frame) > 1 else 0.0
            
            features = {
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
            }
            
            return features
            
        except Exception as e:
            raise ValueError(f"特征提取失败: {str(e)}")
    
    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """将特征字典转为模型输入向量"""
        vec = []
        vec.extend(features.get("mfcc_mean", [0]*13))
        vec.extend(features.get("mfcc_std", [0]*13))
        vec.append(features.get("spectral_centroid", 0))
        vec.append(features.get("spectral_bandwidth", 0))
        vec.append(features.get("spectral_rolloff", 0))
        vec.append(features.get("spectral_flatness", 0))
        vec.append(features.get("zcr", 0))
        vec.append(features.get("rms", 0))
        vec.append(features.get("rms_std", 0))
        vec.append(features.get("duration", 0))
        vec.append(features.get("f0_mean", 0))
        vec.append(features.get("f0_std", 0))
        vec.append(features.get("f0_range", 0))
        vec.append(features.get("voiced_ratio", 0))
        vec.extend(features.get("spectral_contrast", [0]*7))
        return np.array(vec, dtype=np.float32).reshape(1, -1)
    
    def predict(self, audio_path: str) -> Dict:
        features = self.extract_features(audio_path)
        
        if self.model is not None:
            return self._predict_with_model(features)
        else:
            return self._predict_with_rules(features)
    
    def _predict_with_model(self, features: Dict) -> Dict:
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
                "all_probs": probs,
                "features_debug": features  # 调试用
            }
        except Exception as e:
            print(f"模型预测失败: {e}，降级为规则分类器")
            return self._predict_with_rules(features)
    
    def _predict_with_rules(self, features: Dict) -> Dict:
        """
        猫叫分类器 v3 — 决策树优先 + 评分兜底
        
        设计理念:
        1. 先用决策树快速判断"铁证"情况（避免评分被手机降噪干扰）
        2. 评分阶段: centroid权重最大(最不受降噪影响)，zcr/flatness权重降低
        3. food是兜底类别（中间频段的meow），不抢高频和低频
        
        手机录音对特征的影响:
        - AGC → rms差异缩小，不作为主判据
        - 降噪 → zcr和flatness偏低，不能单独作brushing判据
        - 频率响应 → centroid和f0相对稳定，最可靠
        """
        centroid = features.get("spectral_centroid", 0)
        rms = features.get("rms", 0)
        duration = features.get("duration", 0)
        zcr = features.get("zcr", 0)
        bandwidth = features.get("spectral_bandwidth", 0)
        rolloff = features.get("spectral_rolloff", 0)
        flatness = features.get("spectral_flatness", 0)
        f0_mean = features.get("f0_mean", 0)
        f0_std = features.get("f0_std", 0)
        f0_range = features.get("f0_range", 0)
        voiced_ratio = features.get("voiced_ratio", 0)
        
        # ============================================================
        # 第一层：决策树快速判断（铁证级别，不进评分）
        # 这些组合是猫叫声学研究中区分度极高的特征
        # ============================================================
        
        # 规则1: 呼噜 — centroid极低 + flatness极低
        # 真正的呼噜centroid在500-1500Hz，flatness < 0.01
        # 手机降噪只能让flatness变低，但不会让centroid降到1000以下
        if centroid < 1200 and flatness < 0.01:
            return {
                "intent": "brushing",
                "confidence": 0.90,
                "all_probs": {"brushing": 0.90, "food": 0.04, "isolation": 0.02, 
                              "happy": 0.02, "angry": 0.01, "pain": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: brushing (centroid<1200 + flatness<0.01)"
            }
        
        # 规则2: 痛苦howl — 非常长 + 高centroid + f0剧烈波动
        # 放在angry前面：因为痛苦的叫声zcr也高，但痛苦有长时+f0波动特征
        if duration > 2.0 and centroid > 2500 and f0_std > 60:
            return {
                "intent": "pain",
                "confidence": 0.82,
                "all_probs": {"pain": 0.82, "isolation": 0.08, "angry": 0.05,
                              "food": 0.03, "happy": 0.01, "brushing": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: pain (duration>2.0 + centroid>2500 + f0_std>60)"
            }
        
        # 规则3: 哈气hiss — 噪声信号
        # hiss的特征: 高zcr(噪声) + 高centroid 或 高flatness
        # ★ 关键: 纯meow虽然zcr可能>0.10，但会有f0波动，hiss没有f0
        # 所以 angry = 高zcr/flatness + 无f0波动(或极短)
        angry_signal = zcr > 0.20 or flatness > 0.15  # 强噪声信号
        angry_weak = (zcr > 0.12 or flatness > 0.08) and centroid > 2500  # 降噪后弱信号+高频
        not_isolation = f0_std < 30 or duration < 0.8  # 不是焦虑(焦虑有f0波动且长)
        
        if (angry_signal or (angry_weak and not_isolation)):
            return {
                "intent": "angry",
                "confidence": 0.85,
                "all_probs": {"angry": 0.85, "isolation": 0.06, "pain": 0.04,
                              "food": 0.03, "happy": 0.01, "brushing": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: angry (noise signal)"
            }
        
        # 规则4: 焦虑/害怕 — 高centroid + 长时 + f0波动大
        # 焦虑猫叫的音高会剧烈变化，这是降噪无法消除的
        if centroid > 2500 and duration > 1.0 and (f0_std > 50 or f0_range > 200):
            return {
                "intent": "isolation",
                "confidence": 0.85,
                "all_probs": {"isolation": 0.85, "angry": 0.06, "pain": 0.04,
                              "food": 0.03, "happy": 0.01, "brushing": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: isolation (centroid>2500 + duration>1.0 + f0波动)"
            }
        
        # 规则5: 极短chirp — 非常短 + 中频
        if duration < 0.35 and 1000 < centroid < 3500:
            return {
                "intent": "happy",
                "confidence": 0.80,
                "all_probs": {"happy": 0.80, "food": 0.10, "isolation": 0.05,
                              "brushing": 0.02, "angry": 0.02, "pain": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: happy (duration<0.35s + mid-centroid)"
            }
        
        # ============================================================
        # 第二层：评分系统（决策树未命中时）
        # 核心原则: centroid权重最大，food不抢高频/低频区间
        # ============================================================
        scores = {label: 0.0 for label in self.LABELS}
        
        # --- BRUSHING (评分兜底) ---
        # centroid极低是唯一强信号
        if centroid < 1200:
            scores["brushing"] += 6
        elif centroid < 1800:
            scores["brushing"] += 2
        # flatness极低辅助
        if flatness < 0.01:
            scores["brushing"] += 4
        elif flatness < 0.03:
            scores["brushing"] += 1
        # f0极低辅助
        if 0 < f0_mean < 200:
            scores["brushing"] += 3
        elif 0 < f0_mean < 400:
            scores["brushing"] += 1
        # rolloff低辅助
        if rolloff < 2000:
            scores["brushing"] += 2
        
        # --- FOOD (评分兜底，只占中间频段) ---
        # ★ 关键: centroid > 2200时不给food分（那是isolation/angry区间）
        if 1000 < centroid < 2200:
            scores["food"] += 5  # 中频=meow典型范围
        elif 800 < centroid < 2600:
            scores["food"] += 2  # 略宽但权重低
        
        if 0.3 < duration < 1.5:
            scores["food"] += 2
        elif duration < 2.5:
            scores["food"] += 0.5
        
        # 中等谐波性
        if 0.01 < flatness < 0.15:
            scores["food"] += 1
        
        # 中等zcr
        if 0.03 < zcr < 0.15:
            scores["food"] += 0.5
        
        # rolloff中等
        if 2000 < rolloff < 4000:
            scores["food"] += 1
        
        # --- ISOLATION (评分兜底) ---
        # centroid高 + f0波动
        if centroid > 3500:
            scores["isolation"] += 7
        elif centroid > 2800:
            scores["isolation"] += 5
        elif centroid > 2200:
            scores["isolation"] += 3
        elif centroid > 1800:
            scores["isolation"] += 1
        
        if duration > 1.5:
            scores["isolation"] += 3
        elif duration > 0.8:
            scores["isolation"] += 1.5
        
        # f0波动是强信号（不受降噪影响）
        if f0_std > 80:
            scores["isolation"] += 4
        elif f0_std > 40:
            scores["isolation"] += 2
        
        if f0_range > 300:
            scores["isolation"] += 2
        elif f0_range > 150:
            scores["isolation"] += 1
        
        # 高rolloff辅助
        if rolloff > 5000:
            scores["isolation"] += 2
        elif rolloff > 3500:
            scores["isolation"] += 1
        
        # --- HAPPY (评分兜底) ---
        if duration < 0.5:
            scores["happy"] += 4
        elif duration < 0.8:
            scores["happy"] += 2
        
        if 1200 < centroid < 2800:
            scores["happy"] += 1.5
        
        if f0_range > 100 and duration < 0.8:
            scores["happy"] += 2
        
        # --- ANGRY (评分兜底) ---
        # zcr和flatness是主信号（即使被降噪压低，仍有一定区分度）
        if zcr > 0.20:
            scores["angry"] += 6
        elif zcr > 0.15:
            scores["angry"] += 3
        elif zcr > 0.10:
            scores["angry"] += 1
        
        if flatness > 0.15:
            scores["angry"] += 5
        elif flatness > 0.08:
            scores["angry"] += 2
        elif flatness > 0.05:
            scores["angry"] += 0.5
        
        # centroid极高也是angry信号
        if centroid > 4000:
            scores["angry"] += 3
        elif centroid > 3000:
            scores["angry"] += 1.5
        
        # rolloff极高
        if rolloff > 6000:
            scores["angry"] += 2
        elif rolloff > 4500:
            scores["angry"] += 1
        
        # --- PAIN (评分兜底) ---
        if duration > 2.5:
            scores["pain"] += 5
        elif duration > 1.5:
            scores["pain"] += 3
        elif duration > 1.0:
            scores["pain"] += 1
        
        if f0_std > 100:
            scores["pain"] += 4
        elif f0_std > 60:
            scores["pain"] += 2
        
        if f0_range > 400:
            scores["pain"] += 2
        elif f0_range > 200:
            scores["pain"] += 1
        
        if centroid > 2500:
            scores["pain"] += 1
        
        # ============================================================
        # Softmax 转概率
        # ============================================================
        score_array = np.array([scores[k] for k in self.LABELS])
        
        # 最低分保障：没有强信号时默认food（一般meow）
        if np.max(score_array) < 2:
            scores["food"] += 2
            score_array = np.array([scores[k] for k in self.LABELS])
        
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
            "all_probs": probs,
            "features_debug": features,
            "rule_hit": "scoring_fallback"
        }


# 全局分类器实例
classifier = CatSoundClassifier()
