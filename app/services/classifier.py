"""
猫叫分类器服务 v4
- v3基础上修复：用f0_std做呼噜否决条件
- 核心发现：真实呼噜f0几乎不变(std<20)，任何f0剧烈波动的声音都不该判brushing
- 新增food决策树规则：中频centroid+有声+非噪声=讨食meow
- isolation评分的f0加分只在centroid>2200时生效（中频meow的f0波动不算焦虑）
- voiced_ratio和f0_mean作为food辅助信号
"""
import os
import subprocess
import json
import numpy as np
import librosa
from typing import Dict, Optional


class CatSoundClassifier:
    """
    猫叫声分类器 v4
    
    v4 关键修复（基于真实猫叫数据分析）:
    1. ★ f0_std作为呼噜否决条件：真实呼噜f0_std<20，f0_std>30绝不判brushing
       - 驱虫叫声flatness=0.006像呼噜，但f0_std=94.6，绝对不是呼噜
    2. 新增food决策树规则：中频centroid+有声+非噪声=讨食meow
    3. isolation评分的f0加分限制在centroid>2200（中频meow的f0波动不算焦虑）
    4. voiced_ratio和f0_mean作为food辅助信号
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
        猫叫分类器 v4 — 决策树优先 + 评分兜底
        
        v4 vs v3 核心改动:
        1. ★ brushing决策树增加f0_std否决：f0_std>30绝不判brushing
           - 真实呼噜f0几乎不变(std<20)，驱虫叫声f0_std=94.6绝不是呼噜
        2. 新增food决策树：中频centroid+有声+非噪声=meow
        3. isolation评分f0加分限制centroid>2200（中频meow的f0波动不算焦虑）
        4. brushing评分增加f0_std否决扣分
        5. food评分增加voiced_ratio和f0_mean辅助
        
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
        # ============================================================
        
        # 规则1: 呼噜 — centroid极低 + flatness极低 + f0_std极低
        # ★ v4核心修复：真实呼噜f0几乎不变(std<20)，f0剧烈波动的绝不是呼噜
        # 手机降噪可能让flatness变低，但不会产生稳定的f0
        if centroid < 1200 and flatness < 0.01 and f0_std < 30:
            return {
                "intent": "brushing",
                "confidence": 0.90,
                "all_probs": {"brushing": 0.90, "food": 0.04, "isolation": 0.02, 
                              "happy": 0.02, "angry": 0.01, "pain": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: brushing (centroid<1200 + flatness<0.01 + f0_std<30)"
            }
        
        # 规则2: 痛苦/哀鸣 — 长时 + f0剧烈波动
        # ★ v4: 去掉centroid>2500限制，用f0_std/f0_mean比率区分pain和food
        # 驱虫叫声: duration=3.83s + f0_std=94.6 + ratio=0.243 → pain
        # 讨食meow: duration=2.37s + f0_std=82 + ratio=0.100 → food(不误判)
        # 原理：pain的音高波动比例远大于food（哀鸣的音高剧烈抖动vs meow的温和变化）
        f0_variation_ratio = f0_std / max(f0_mean, 1) if f0_mean > 0 else 0
        is_long_pain = duration > 3.0 and f0_std > 40  # 非常长的哀鸣
        is_intense_pain = duration > 2.0 and f0_std > 60 and f0_variation_ratio > 0.15  # 中等长度但波动剧烈
        
        if is_long_pain or is_intense_pain:
            return {
                "intent": "pain",
                "confidence": 0.82,
                "all_probs": {"pain": 0.82, "isolation": 0.08, "angry": 0.05,
                              "food": 0.03, "happy": 0.01, "brushing": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: pain (long distress or intense f0 variation)"
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
        
        # ★ 规则5 (v4新增): 讨食meow — 中频centroid + 有声调 + 非噪声
        # 这是最常见的猫叫声类型，v3漏了决策树规则导致进评分被isolation抢走
        is_mid_freq = 800 < centroid < 2400
        is_voiced = voiced_ratio > 0.25
        is_not_noisy = zcr < 0.22 and flatness < 0.20
        
        if is_mid_freq and is_voiced and is_not_noisy:
            if duration > 0.5:  # 稍长的meow = 讨食
                return {
                    "intent": "food",
                    "confidence": 0.82,
                    "all_probs": {"food": 0.82, "isolation": 0.07, "happy": 0.05,
                                  "brushing": 0.03, "angry": 0.02, "pain": 0.01},
                    "features_debug": features,
                    "rule_hit": "decision_tree: food (mid-centroid + voiced + not-noisy)"
                }
            else:  # 极短的meow = 开心
                return {
                    "intent": "happy",
                    "confidence": 0.78,
                    "all_probs": {"happy": 0.78, "food": 0.12, "isolation": 0.05,
                                  "brushing": 0.02, "angry": 0.02, "pain": 0.01},
                    "features_debug": features,
                    "rule_hit": "decision_tree: happy (mid-centroid + voiced + short)"
                }
        
        # 规则6: 极短chirp — 非常短 + 中频
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
        # v4关键修改: brushing评分增加f0_std否决，isolation的f0加分限centroid>2200
        # ============================================================
        scores = {label: 0.0 for label in self.LABELS}
        
        # --- BRUSHING (评分兜底) ---
        # ★ v4: f0_std>30时brushing大幅扣分（呼噜f0必须稳定）
        if f0_std < 20:
            scores["brushing"] += 5  # f0很稳，强呼噜信号
        elif f0_std < 30:
            scores["brushing"] += 2  # f0较稳
        # f0_std>30不加分，f0_std>50扣分
        if f0_std > 50:
            scores["brushing"] -= 4  # f0剧烈波动，绝不可能是呼噜
        elif f0_std > 30:
            scores["brushing"] -= 2
        
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
        
        # --- FOOD (评分兜底，v4加强) ---
        # ★ v4: centroid在中频时food权重更高
        if 800 < centroid < 2400:
            scores["food"] += 7  # meow核心频段，v4提升
        elif 600 < centroid < 2800:
            scores["food"] += 3
        
        if 0.3 < duration < 2.5:
            scores["food"] += 2
        
        # v4: 放宽flatness范围
        if 0.01 < flatness < 0.18:
            scores["food"] += 1.5
        
        # v4: 放宽zcr范围
        if 0.03 < zcr < 0.18:
            scores["food"] += 1
        
        if 2000 < rolloff < 4500:
            scores["food"] += 1
        
        # ★ v4新增: voiced_ratio中等 = 有声调的meow
        if voiced_ratio > 0.2:
            scores["food"] += 2
        
        # ★ v4新增: f0_mean在猫叫典型范围(300-1200Hz)
        if 300 < f0_mean < 1200:
            scores["food"] += 2
        
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
        
        # ★ v4核心修改: f0波动只在centroid>2200时才给isolation加分
        # 中频meow(centroid 800-2400)也有f0波动，但那是food不是isolation
        if centroid > 2200:
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
