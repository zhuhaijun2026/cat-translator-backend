"""
猫叫分类器服务 v5
- v4基础上修复：f0_std用IQR异常值过滤后的robust版本
- 核心发现1：pyin在间歇性猫叫过渡帧产生八度跳跃错误(~203Hz vs 真实~440Hz)
  23%帧是八度错误，虚增f0_std从6.5→94.6，导致驱虫声误判pain
- 核心发现2：猫驱虫/驱赶飞虫的chattering声学特征像呼噜(centroid低、flatness低、f0稳)
  但实际上有大量中高频能量(meow成分)，纯呼噜500Hz以上能量<5%，chattering>30%
  → 新增high_freq_energy_ratio特征，brushing决策树增加hf_ratio<0.15限制
- 修复方案：IQR过滤 + hf_ratio区分呼噜与chattering
"""
import os
import subprocess
import json
import numpy as np
import librosa
from typing import Dict, Optional


class CatSoundClassifier:
    """
    猫叫声分类器 v5
    
    v5 vs v4 核心改动:
    1. ★ f0_std改用IQR过滤后的robust版本（消除pyin八度跳跃错误）
       - 驱虫声raw f0_std=94.6 → robust f0_std=6.5，不再误判pain
       - 真正痛苦声的f0变化是真实的、跨多帧的，IQR过滤后仍高
    2. ★ 新增high_freq_energy_ratio特征：500Hz以上频段能量占总能量比例
       - 纯呼噜: <0.05, chattering/驱虫: >0.30, meow: >0.60
       - brushing决策树增加hf_ratio<0.15限制，防止chattering误判brushing
    3. brushing决策树f0_std阈值收紧：<20（robust版更稳定，真实呼噜<5）
    4. pain规则阈值适配robust f0_std：长痛>50，剧烈>40+ratio>0.15
    5. 所有决策树和评分规则统一使用robust版本
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
            print("⚠️ 未加载预训练模型，使用v5评分规则分类器")
    
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
    
    @staticmethod
    def _robust_f0_stats(f0_valid: np.ndarray) -> tuple:
        """
        用IQR过滤f0八度跳跃错误，返回robust统计量
        
        pyin在间歇性猫叫过渡帧会产生八度跳跃：
        - 真实pitch ~440Hz，但过渡帧估计为 ~203Hz（低一个八度）
        - 这些异常帧仅占23%，但把f0_std从6.5虚增到94.6
        
        IQR过滤方法：Q1-1.5*IQR ~ Q3+1.5*IQR范围外的视为异常值
        真正的音高变化（痛苦哀鸣的剧烈波动）跨多帧，不会被过滤
        """
        if len(f0_valid) < 3:
            return (float(f0_valid.mean()) if len(f0_valid) > 0 else 0,
                    float(f0_valid.std()) if len(f0_valid) > 0 else 0,
                    float(f0_valid.max() - f0_valid.min()) if len(f0_valid) > 0 else 0)
        
        q25, q75 = np.percentile(f0_valid, [25, 75])
        iqr = q75 - q25
        
        # IQR过滤：保留 Q1-1.5*IQR ~ Q3+1.5*IQR 范围内的帧
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        filtered = f0_valid[(f0_valid >= lower_bound) & (f0_valid <= upper_bound)]
        
        # 如果过滤后帧太少（<50%），放宽过滤条件
        if len(filtered) < len(f0_valid) * 0.5:
            # 用MAD过滤代替：偏离中位数超过3*MAD的视为异常
            median = np.median(f0_valid)
            mad = np.median(np.abs(f0_valid - median))
            if mad > 0:
                filtered = f0_valid[np.abs(f0_valid - median) <= 3 * mad * 1.4826]
            # 如果还是太少，用原始数据
            if len(filtered) < len(f0_valid) * 0.3:
                filtered = f0_valid
        
        f0_mean = float(np.mean(filtered))
        f0_std = float(np.std(filtered))
        f0_range = float(np.max(filtered) - np.min(filtered))
        
        return f0_mean, f0_std, f0_range
    
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
            
            # 音高提取 — ★ v5: 增加robust统计量
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
                )
                f0_valid = f0[~np.isnan(f0)]
                if len(f0_valid) > 0:
                    # 原始统计量（保留用于对比调试）
                    f0_mean_raw = float(np.mean(f0_valid))
                    f0_std_raw = float(np.std(f0_valid))
                    f0_range_raw = float(np.max(f0_valid) - np.min(f0_valid))
                    voiced_ratio = float(np.sum(voiced_flag) / len(voiced_flag))
                    
                    # ★ v5核心修复：IQR过滤后的robust统计量
                    f0_mean, f0_std, f0_range = self._robust_f0_stats(f0_valid)
                    f0_median = float(np.median(f0_valid))
                else:
                    f0_mean = f0_std = f0_range = 0
                    f0_mean_raw = f0_std_raw = f0_range_raw = 0
                    f0_median = 0
                    voiced_ratio = 0
            except:
                f0_mean = f0_std = f0_range = 0
                f0_mean_raw = f0_std_raw = f0_range_raw = 0
                f0_median = 0
                voiced_ratio = 0
            
            # ★ v5新增: 高频能量比 — 区分纯呼噜与chattering
            # 纯呼噜能量集中在<200Hz，500Hz以上<5%
            # chattering/驱虫声有meow成分，500Hz以上>30%
            try:
                S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
                fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
                total_energy = np.sum(S)
                high_freq_energy = np.sum(S[fft_freqs > 500, :])
                high_freq_energy_ratio = float(high_freq_energy / total_energy) if total_energy > 0 else 0.0
            except:
                high_freq_energy_ratio = 0.0
            
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
                "f0_mean": f0_mean,           # ★ v5: robust mean
                "f0_std": f0_std,             # ★ v5: robust std
                "f0_range": f0_range,         # ★ v5: robust range
                "f0_median": f0_median,       # ★ v5新增
                "f0_mean_raw": f0_mean_raw,   # 调试对比用
                "f0_std_raw": f0_std_raw,     # 调试对比用
                "f0_range_raw": f0_range_raw, # 调试对比用
                "voiced_ratio": voiced_ratio,
                "rms_std": rms_std,
                "high_freq_energy_ratio": high_freq_energy_ratio,  # ★ v5新增
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
                "features_debug": features
            }
        except Exception as e:
            print(f"模型预测失败: {e}，降级为规则分类器")
            return self._predict_with_rules(features)
    
    def _predict_with_rules(self, features: Dict) -> Dict:
        """
        猫叫分类器 v5 — 决策树优先 + 评分兜底
        
        v5 vs v4 核心改动:
        1. ★ f0_std改用IQR过滤后的robust版本
           - pyin八度跳跃错误被过滤：驱虫声94.6→6.5，不再误判pain
           - 真正的pain变化保留：因为跨多帧的真实波动不会被IQR过滤
        2. brushing决策树f0_std<20（robust版更准，真实呼噜f0_std<5）
        3. pain规则阈值适配：长痛robust_f0_std>50，剧烈>40+ratio>0.15
        """
        centroid = features.get("spectral_centroid", 0)
        rms = features.get("rms", 0)
        duration = features.get("duration", 0)
        zcr = features.get("zcr", 0)
        bandwidth = features.get("spectral_bandwidth", 0)
        rolloff = features.get("spectral_rolloff", 0)
        flatness = features.get("spectral_flatness", 0)
        f0_mean = features.get("f0_mean", 0)       # ★ v5: robust
        f0_std = features.get("f0_std", 0)         # ★ v5: robust
        f0_range = features.get("f0_range", 0)     # ★ v5: robust
        f0_median = features.get("f0_median", 0)   # ★ v5新增
        voiced_ratio = features.get("voiced_ratio", 0)
        hf_ratio = features.get("high_freq_energy_ratio", 0)  # ★ v5新增
        
        # ============================================================
        # 第一层：决策树快速判断（铁证级别，不进评分）
        # ============================================================
        
        # 规则1: 呼噜 — centroid极低 + flatness极低 + f0_std极低 + 高频能量极低
        # ★ v5核心: hf_ratio<0.15 区分纯呼噜与chattering
        # 纯呼噜: 500Hz以上能量<5%, chattering/驱虫: >30%, meow: >60%
        if centroid < 1200 and flatness < 0.01 and f0_std < 20 and hf_ratio < 0.15:
            return {
                "intent": "brushing",
                "confidence": 0.90,
                "all_probs": {"brushing": 0.90, "food": 0.04, "isolation": 0.02, 
                              "happy": 0.02, "angry": 0.01, "pain": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: brushing (centroid<1200 + flatness<0.01 + robust_f0_std<20 + hf_ratio<0.15)"
            }
        
        # 规则2: 痛苦/哀鸣 — 长时 + f0剧烈波动（robust版）
        # ★ v5: 使用robust f0_std，阈值调整适配
        # 真正的痛苦哀鸣：f0在多帧间剧烈波动，IQR过滤后仍高
        # 间歇性叫声的八度跳跃：IQR过滤后f0_std很低，不会误触发
        f0_variation_ratio = f0_std / max(f0_mean, 1) if f0_mean > 0 else 0
        is_long_pain = duration > 3.0 and f0_std > 50  # ★ v5: robust, >50
        is_intense_pain = duration > 2.0 and f0_std > 40 and f0_variation_ratio > 0.15  # ★ v5: robust, >40
        
        if is_long_pain or is_intense_pain:
            return {
                "intent": "pain",
                "confidence": 0.82,
                "all_probs": {"pain": 0.82, "isolation": 0.08, "angry": 0.05,
                              "food": 0.03, "happy": 0.01, "brushing": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: pain (long distress or intense f0 variation, robust_f0_std)"
            }
        
        # 规则3: 哈气hiss — 噪声信号
        angry_signal = zcr > 0.20 or flatness > 0.15
        angry_weak = (zcr > 0.12 or flatness > 0.08) and centroid > 2500
        not_isolation = f0_std < 30 or duration < 0.8
        
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
        if centroid > 2500 and duration > 1.0 and (f0_std > 50 or f0_range > 200):
            return {
                "intent": "isolation",
                "confidence": 0.85,
                "all_probs": {"isolation": 0.85, "angry": 0.06, "pain": 0.04,
                              "food": 0.03, "happy": 0.01, "brushing": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: isolation (centroid>2500 + duration>1.0 + f0波动)"
            }
        
        # 规则5: 讨食meow — 中频centroid + 有声调 + 非噪声
        is_mid_freq = 800 < centroid < 2400
        is_voiced = voiced_ratio > 0.25
        is_not_noisy = zcr < 0.22 and flatness < 0.20
        
        if is_mid_freq and is_voiced and is_not_noisy:
            if duration > 0.5:
                return {
                    "intent": "food",
                    "confidence": 0.82,
                    "all_probs": {"food": 0.82, "isolation": 0.07, "happy": 0.05,
                                  "brushing": 0.03, "angry": 0.02, "pain": 0.01},
                    "features_debug": features,
                    "rule_hit": "decision_tree: food (mid-centroid + voiced + not-noisy)"
                }
            else:
                return {
                    "intent": "happy",
                    "confidence": 0.78,
                    "all_probs": {"happy": 0.78, "food": 0.12, "isolation": 0.05,
                                  "brushing": 0.02, "angry": 0.02, "pain": 0.01},
                    "features_debug": features,
                    "rule_hit": "decision_tree: happy (mid-centroid + voiced + short)"
                }
        
        # 规则6: 极短chirp
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
        # v5: 全部使用robust f0_std
        # ============================================================
        scores = {label: 0.0 for label in self.LABELS}
        
        # --- BRUSHING (评分兜底) ---
        # ★ v5: robust f0_std更准确，呼噜<5，非呼噜>10
        # ★ v5: hf_ratio>0.15说明有meow成分，不是纯呼噜
        if f0_std < 10:
            scores["brushing"] += 5  # 极稳定f0，强呼噜信号
        elif f0_std < 20:
            scores["brushing"] += 2
        if f0_std > 30:
            scores["brushing"] -= 4  # robust版还高说明真有波动
        elif f0_std > 20:
            scores["brushing"] -= 2
        
        # ★ v5新增: 高频能量比 — chattering/驱虫有大量中高频能量
        # 这是最关键的呼噜/chattering区分信号，权重必须足够大
        # 纯呼噜: hf<0.05, chattering: 0.30-0.50, meow: >0.60
        if hf_ratio < 0.05:
            scores["brushing"] += 6  # 几乎无中高频，纯呼噜
        elif hf_ratio < 0.15:
            scores["brushing"] += 2  # 中高频很少
        if hf_ratio > 0.30:
            scores["brushing"] -= 12  # ★ 强否决：中高频很多，绝不是呼噜
        elif hf_ratio > 0.15:
            scores["brushing"] -= 5  # 中高频较多，可疑
        
        if centroid < 1200:
            scores["brushing"] += 6
        elif centroid < 1800:
            scores["brushing"] += 2
        if flatness < 0.01:
            scores["brushing"] += 4
        elif flatness < 0.03:
            scores["brushing"] += 1
        if 0 < f0_mean < 200:
            scores["brushing"] += 3
        elif 0 < f0_mean < 400:
            scores["brushing"] += 1
        if rolloff < 2000:
            scores["brushing"] += 2
        
        # --- FOOD (评分兜底) ---
        if 800 < centroid < 2400:
            scores["food"] += 7
        elif 600 < centroid < 2800:
            scores["food"] += 3
        
        # ★ v5: centroid极低但hf_ratio高 → chattering(驱虫)也走food路径
        # chattering的centroid被低频呼噜成分拉低，但hf_ratio揭示meow成分
        if centroid < 800 and hf_ratio > 0.25:
            scores["food"] += 5  # 低centroid但高频能量高 = chattering
        
        if 0.3 < duration < 2.5:
            scores["food"] += 2
        
        if 0.01 < flatness < 0.18:
            scores["food"] += 1.5
        
        if 0.03 < zcr < 0.18:
            scores["food"] += 1
        
        if 2000 < rolloff < 4500:
            scores["food"] += 1
        
        if voiced_ratio > 0.2:
            scores["food"] += 2
        
        if 300 < f0_mean < 1200:
            scores["food"] += 2
        
        # ★ v5新增: hf_ratio高说明有meow成分（chattering/驱虫也走food路径）
        if hf_ratio > 0.30:
            scores["food"] += 4  # 大量meow频段能量
        elif hf_ratio > 0.15:
            scores["food"] += 2
        
        # --- ISOLATION (评分兜底) ---
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
        
        # f0波动只在centroid>2200时才给isolation加分
        if centroid > 2200:
            if f0_std > 80:
                scores["isolation"] += 4
            elif f0_std > 40:
                scores["isolation"] += 2
            if f0_range > 300:
                scores["isolation"] += 2
            elif f0_range > 150:
                scores["isolation"] += 1
        
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
        
        if centroid > 4000:
            scores["angry"] += 3
        elif centroid > 3000:
            scores["angry"] += 1.5
        
        if rolloff > 6000:
            scores["angry"] += 2
        elif rolloff > 4500:
            scores["angry"] += 1
        
        # --- PAIN (评分兜底) ---
        # ★ v5: 使用robust f0_std，阈值适配
        if duration > 2.5:
            scores["pain"] += 5
        elif duration > 1.5:
            scores["pain"] += 3
        elif duration > 1.0:
            scores["pain"] += 1
        
        if f0_std > 80:
            scores["pain"] += 4
        elif f0_std > 50:
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
