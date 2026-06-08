"""
猫叫分类器服务 v6.2
- v6.1基础上修复不满抱怨声被误判food
- v6.1修了反复meow讨食被误判pain，但不满抱怨声centroid在food范围内被food决策树截获
- 核心发现：不满抱怨 voiced_ratio>0.6(连续发声) vs food voiced_ratio 0.3-0.6(断续meow)
- v6.2修复：
  1. 新增"连续抱怨"决策树规则：低centroid+高f0_std+高voiced_ratio → angry
  2. food决策树增加非连续抱怨条件
  3. angry评分增加连续抱怨加分
  这是猫独有的混合发声：同时发出低频呼噜和中高频meow（驱赶猎物时特有）
- 区分food vs chattering的关键：
  food: centroid>1200, flatness>0.05, f0_std>20 (典型meow)
  chattering: centroid<1500, flatness<0.02, f0_std<30 (呼噜+meow混合)
"""
import os
import subprocess
import json
import numpy as np
import librosa
from typing import Dict, Optional


class CatSoundClassifier:
    """
    猫叫声分类器 v6
    
    v6 vs v5 核心改动:
    1. ★ 新增chattering类别 — 猫驱赶/追逐猎物的特有声音
       声学指纹：低centroid(<1500) + 低flatness(<0.02) + 高hf_ratio(>0.20) + 低f0_std(<30)
       这组合唯一标识chattering：呼噜的低频稳定性 + meow的中高频能量
    2. 决策树优先级调整：chattering规则插在brushing之后、food之前
       避免chattering被food规则截获
    3. 评分系统增加chattering评分项
    """
    
    LABELS = ["brushing", "food", "isolation", "happy", "angry", "pain", "chattering"]
    
    # 类别中文映射
    LABEL_CN = {
        "brushing": "满足(呼噜)",
        "food": "急切(讨食)",
        "isolation": "焦虑(呼唤)",
        "happy": "开心(互动)",
        "angry": "不满(威吓)",
        "pain": "痛苦(哀鸣)",
        "chattering": "兴奋(追逐)"
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("⚠️ 未加载预训练模型，使用v6评分规则分类器")
    
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
        """
        if len(f0_valid) < 3:
            return (float(f0_valid.mean()) if len(f0_valid) > 0 else 0,
                    float(f0_valid.std()) if len(f0_valid) > 0 else 0,
                    float(f0_valid.max() - f0_valid.min()) if len(f0_valid) > 0 else 0)
        
        q25, q75 = np.percentile(f0_valid, [25, 75])
        iqr = q75 - q25
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        filtered = f0_valid[(f0_valid >= lower_bound) & (f0_valid <= upper_bound)]
        
        if len(filtered) < len(f0_valid) * 0.5:
            median = np.median(f0_valid)
            mad = np.median(np.abs(f0_valid - median))
            if mad > 0:
                filtered = f0_valid[np.abs(f0_valid - median) <= 3 * mad * 1.4826]
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
            
            # 音高提取
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
                )
                f0_valid = f0[~np.isnan(f0)]
                if len(f0_valid) > 0:
                    f0_mean_raw = float(np.mean(f0_valid))
                    f0_std_raw = float(np.std(f0_valid))
                    f0_range_raw = float(np.max(f0_valid) - np.min(f0_valid))
                    voiced_ratio = float(np.sum(voiced_flag) / len(voiced_flag))
                    
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
            
            # ★ v6.1新增: 帧间f0变化中位数 — 区分多段meow与持续哀鸣
            # 多段meow: 段间音高跳变大(全局f0_std高)但段内帧间变化极小(median<10)
            # 持续哀鸣: 逐帧剧烈变化(median>20)
            try:
                f0_series = f0.copy()
                f0_series[np.isnan(f0_series)] = 0
                frame_diffs = []
                for i in range(1, len(f0_series)):
                    if f0_series[i] > 0 and f0_series[i-1] > 0:
                        frame_diffs.append(abs(f0_series[i] - f0_series[i-1]))
                f0_frame_diff_median = float(np.median(frame_diffs)) if len(frame_diffs) > 0 else 0.0
            except:
                f0_frame_diff_median = 0.0
            
            # 高频能量比
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
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "f0_range": f0_range,
                "f0_median": f0_median,
                "f0_mean_raw": f0_mean_raw,
                "f0_std_raw": f0_std_raw,
                "f0_range_raw": f0_range_raw,
                "voiced_ratio": voiced_ratio,
                "rms_std": rms_std,
                "high_freq_energy_ratio": high_freq_energy_ratio,
                "f0_frame_diff_median": f0_frame_diff_median,  # ★ v6.1新增
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
        猫叫分类器 v6 — 决策树优先 + 评分兜底
        
        v6 vs v5 核心改动:
        1. ★ 新增chattering决策树规则（插在brushing之后、food之前）
           chattering声学指纹：低centroid + 低flatness + 高hf_ratio + 低f0_std
           这是猫驱赶/追逐猎物的特有混合发声：同时发出呼噜低频和meow中高频
        2. food决策树增加flatness>0.02前置条件，排除chattering误入
        3. 评分系统增加chattering评分项
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
        f0_median = features.get("f0_median", 0)
        voiced_ratio = features.get("voiced_ratio", 0)
        hf_ratio = features.get("high_freq_energy_ratio", 0)
        f0_frame_diff_median = features.get("f0_frame_diff_median", 0)  # ★ v6.1新增
        
        # ============================================================
        # 第一层：决策树快速判断（铁证级别，不进评分）
        # ============================================================
        
        # 规则1: 呼噜 — centroid极低 + flatness极低 + f0_std极低 + 高频能量极低
        if centroid < 1200 and flatness < 0.01 and f0_std < 20 and hf_ratio < 0.15:
            return {
                "intent": "brushing",
                "confidence": 0.90,
                "all_probs": {"brushing": 0.90, "food": 0.03, "isolation": 0.02, 
                              "happy": 0.01, "angry": 0.01, "pain": 0.01, "chattering": 0.02},
                "features_debug": features,
                "rule_hit": "decision_tree: brushing (centroid<1200 + flatness<0.01 + robust_f0_std<20 + hf_ratio<0.15)"
            }
        
        # ★ 规则2: chattering(驱赶/追逐) — v6新增核心规则
        # 声学指纹：低centroid(像呼噜) + 低flatness(像呼噜) + 高hf_ratio(不像呼噜!) + 低f0_std(稳定)
        # 这是猫chattering/chirping的独特混合发声：同时发出低频呼噜成分和中高频meow成分
        # 真实数据验证：
        #   驱赶飞虫: centroid=714, flatness=0.006, hf_ratio=0.473, f0_std=4.6, voiced=0.98
        #   驱虫声:   centroid=1347, flatness=0.006, hf_ratio=0.734, f0_std=6.5, voiced=0.76
        #   讨食meow: centroid=1516, flatness=0.132, hf_ratio=0.917, f0_std=40.9, voiced=0.41 ← 不会误判
        #   纯呼噜:   hf_ratio<0.05 ← 不会误判(已被规则1截获)
        
        # 主规则：典型chattering
        is_low_centroid = centroid < 1500
        is_tonal = flatness < 0.02  # 极低flatness = 高度调性(像呼噜不像噪声)
        has_meow_component = hf_ratio > 0.20  # 中高频能量显著(不像纯呼噜)
        is_steady_pitch = f0_std < 30  # f0稳定(呼噜成分的稳定性)
        is_continuously_voiced = voiced_ratio > 0.4
        
        if is_low_centroid and is_tonal and has_meow_component and is_steady_pitch and is_continuously_voiced:
            return {
                "intent": "chattering",
                "confidence": 0.92,
                "all_probs": {"chattering": 0.92, "food": 0.03, "brushing": 0.02,
                              "isolation": 0.01, "happy": 0.01, "angry": 0.005, "pain": 0.005},
                "features_debug": features,
                "rule_hit": "decision_tree: chattering (centroid<1500 + flatness<0.02 + hf_ratio>0.20 + robust_f0_std<30 + voiced>0.4)"
            }
        
        # 拓展规则：centroid稍高但仍是chattering模式
        # 有些chattering的meow成分更强，把centroid推到1500-2000
        # 但flatness仍然极低 + hf_ratio高 + f0_std低 = chattering而非food
        if 1500 <= centroid < 2000 and flatness < 0.02 and hf_ratio > 0.40 and f0_std < 25 and voiced_ratio > 0.4:
            return {
                "intent": "chattering",
                "confidence": 0.85,
                "all_probs": {"chattering": 0.85, "food": 0.06, "isolation": 0.04,
                              "brushing": 0.02, "happy": 0.02, "angry": 0.005, "pain": 0.005},
                "features_debug": features,
                "rule_hit": "decision_tree: chattering (centroid 1500-2000 + flatness<0.02 + hf_ratio>0.40 + f0_std<25)"
            }
        
        # 规则3: 痛苦/哀鸣 — 长时 + f0剧烈波动 + 高centroid + 帧间变化大
        # ★ v6.1: 用f0_frame_diff_median区分多段meow与持续哀鸣
        #   多段meow(讨食): 全局f0_std高(多段不同音高) 但帧间变化中位数极小(<10)
        #   持续哀鸣(痛苦): 逐帧f0剧烈变化(帧间变化中位数>15)
        #   烦躁(不满): centroid<1500 + f0_std高 但不是持续哀鸣
        f0_variation_ratio = f0_std / max(f0_mean, 1) if f0_mean > 0 else 0
        is_truly_wailing = f0_frame_diff_median > 15  # ★ v6.1: 真正的逐帧f0抖动
        is_long_pain = duration > 3.0 and f0_std > 50 and centroid > 1500 and is_truly_wailing
        is_intense_pain = duration > 2.0 and f0_std > 40 and f0_variation_ratio > 0.15 and centroid > 1500 and is_truly_wailing
        
        # ★ v6.1: 烦躁/不满 — 长时 + 低centroid + 真正逐帧抖动(非多段meow)
        is_irritated = duration > 2.0 and f0_std > 40 and centroid < 1500 and is_truly_wailing
        
        # ★ v6.2: 连续抱怨 — 低centroid + 高f0_std + 高voiced_ratio(几乎连续发声)
        # 不满抱怨的猫几乎不停地在叫(voiced_ratio>0.6)，不是food那种断续meow(voiced_ratio 0.3-0.6)
        # 真实数据：不满抱怨 centroid=1334, f0_std=65.5, voiced_ratio=0.91, fdm=5.5
        # 对比：  讨食meow  centroid=1516, f0_std=40.9, voiced_ratio=0.41
        #         讨食反复  centroid=1722, f0_std=344,  voiced_ratio=0.56
        #         烦躁声    centroid=1054, f0_std=115.4, voiced_ratio=0.64
        # voiced_ratio是关键区分：food断续叫(0.3-0.6) vs 抱怨连续叫(>0.6)
        # ★ 重要排除：反复meow(高f0_std>150 + 低fdm<10)不是抱怨，是food！
        #   反复meow的f0_std极高是因为多段meow音高不同，但每段内部f0很稳定(fdm低)
        #   连续抱怨的f0_std适中(40-150)，是持续发声+适度抖动
        is_repeated_meow = f0_std > 150 and f0_frame_diff_median < 10
        is_complaining = (duration > 1.5 and f0_std > 40 and centroid < 1500 
                         and voiced_ratio > 0.6 and f0_frame_diff_median < 15
                         and not is_repeated_meow)
        
        if is_irritated or is_complaining:
            return {
                "intent": "angry",
                "confidence": 0.82,
                "all_probs": {"angry": 0.82, "pain": 0.06, "isolation": 0.04,
                              "food": 0.03, "chattering": 0.02, "happy": 0.02, "brushing": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: angry/irritated (long + high f0_std + low centroid)"
            }
        
        if is_long_pain or is_intense_pain:
            return {
                "intent": "pain",
                "confidence": 0.82,
                "all_probs": {"pain": 0.82, "isolation": 0.07, "angry": 0.04,
                              "food": 0.03, "chattering": 0.02, "happy": 0.01, "brushing": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: pain (long + high f0_std + centroid>1500)"
            }
        
        # 规则4: 哈气hiss — 噪声信号
        angry_signal = zcr > 0.20 or flatness > 0.15
        angry_weak = (zcr > 0.12 or flatness > 0.08) and centroid > 2500
        not_isolation = f0_std < 30 or duration < 0.8
        
        if (angry_signal or (angry_weak and not_isolation)):
            return {
                "intent": "angry",
                "confidence": 0.85,
                "all_probs": {"angry": 0.85, "isolation": 0.05, "pain": 0.04,
                              "food": 0.02, "chattering": 0.02, "happy": 0.01, "brushing": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: angry (noise signal)"
            }
        
        # 规则5: 焦虑/害怕 — 高centroid + 长时 + f0波动大
        if centroid > 2500 and duration > 1.0 and (f0_std > 50 or f0_range > 200):
            return {
                "intent": "isolation",
                "confidence": 0.85,
                "all_probs": {"isolation": 0.85, "angry": 0.05, "pain": 0.04,
                              "food": 0.02, "chattering": 0.02, "happy": 0.01, "brushing": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: isolation (centroid>2500 + duration>1.0 + f0波动)"
            }
        
        # 规则6: 讨食meow — 中高频centroid + 有声调 + 非噪声 + ★flatness>0.02(排除chattering)
        # ★ v6.1: centroid下限1100，排除低centroid烦躁声(centroid~1054)
        # 真正讨食的meow centroid通常>1200，烦躁反复叫的centroid<1100
        is_mid_freq = 1100 < centroid < 2400  # ★ v6.1: 800→1100
        is_voiced = voiced_ratio > 0.25
        is_not_noisy = zcr < 0.22 and flatness < 0.20
        is_not_chattering_pattern = flatness > 0.02 or hf_ratio < 0.20 or f0_std > 30  # ★ v6: 排除chattering
        # ★ v6.2: 排除连续抱怨 — 低centroid+高f0_std+高voiced_ratio = 不满/抱怨，不是food
        # 但反复meow(高f0_std>150+低fdm<10)不是抱怨，不能排除
        is_repeated_meow_check = f0_std > 150 and f0_frame_diff_median < 10
        is_not_complaining = centroid >= 1500 or f0_std <= 40 or voiced_ratio <= 0.6 or is_repeated_meow_check
        
        if is_mid_freq and is_voiced and is_not_noisy and is_not_chattering_pattern and is_not_complaining:
            if duration > 0.5:
                return {
                    "intent": "food",
                    "confidence": 0.82,
                    "all_probs": {"food": 0.82, "isolation": 0.06, "happy": 0.04,
                                  "chattering": 0.03, "brushing": 0.02, "angry": 0.02, "pain": 0.01},
                    "features_debug": features,
                    "rule_hit": "decision_tree: food (mid-centroid + voiced + not-noisy + not-chattering-pattern)"
                }
            else:
                return {
                    "intent": "happy",
                    "confidence": 0.78,
                    "all_probs": {"happy": 0.78, "food": 0.10, "isolation": 0.04,
                                  "chattering": 0.03, "brushing": 0.02, "angry": 0.02, "pain": 0.01},
                    "features_debug": features,
                    "rule_hit": "decision_tree: happy (mid-centroid + voiced + short)"
                }
        
        # 规则7: 极短chirp
        if duration < 0.35 and 1000 < centroid < 3500:
            return {
                "intent": "happy",
                "confidence": 0.80,
                "all_probs": {"happy": 0.80, "food": 0.08, "isolation": 0.04,
                              "chattering": 0.03, "brushing": 0.02, "angry": 0.02, "pain": 0.01},
                "features_debug": features,
                "rule_hit": "decision_tree: happy (duration<0.35s + mid-centroid)"
            }
        
        # ============================================================
        # 第二层：评分系统（决策树未命中时）
        # ============================================================
        scores = {label: 0.0 for label in self.LABELS}
        
        # --- CHATTERING (评分兜底) ---
        # 核心：低centroid + 低flatness + 高hf_ratio + 低f0_std
        if centroid < 1500:
            scores["chattering"] += 4
        elif centroid < 2000:
            scores["chattering"] += 2
        
        if flatness < 0.02:
            scores["chattering"] += 4
        elif flatness < 0.05:
            scores["chattering"] += 1
        
        if hf_ratio > 0.30:
            scores["chattering"] += 5
        elif hf_ratio > 0.20:
            scores["chattering"] += 3
        elif hf_ratio > 0.15:
            scores["chattering"] += 1
        
        if f0_std < 20:
            scores["chattering"] += 3
        elif f0_std < 30:
            scores["chattering"] += 1
        
        if voiced_ratio > 0.6:
            scores["chattering"] += 2
        elif voiced_ratio > 0.4:
            scores["chattering"] += 1
        
        # chattering否决条件
        if flatness > 0.10:
            scores["chattering"] -= 5  # flatness高 = 噪声，不是chattering
        if f0_std > 40:
            scores["chattering"] -= 4  # f0不稳定，不是chattering
        if hf_ratio < 0.10:
            scores["chattering"] -= 3  # 高频太少，是纯呼噜不是chattering
        
        # --- BRUSHING (评分兜底) ---
        if f0_std < 10:
            scores["brushing"] += 5
        elif f0_std < 20:
            scores["brushing"] += 2
        if f0_std > 30:
            scores["brushing"] -= 4
        elif f0_std > 20:
            scores["brushing"] -= 2
        
        if hf_ratio < 0.05:
            scores["brushing"] += 6
        elif hf_ratio < 0.15:
            scores["brushing"] += 2
        if hf_ratio > 0.30:
            scores["brushing"] -= 12
        elif hf_ratio > 0.15:
            scores["brushing"] -= 5
        
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
        if 1100 < centroid < 2400:  # ★ v6.1: 800→1100
            scores["food"] += 7
        elif 800 < centroid < 2800:  # 降级范围
            scores["food"] += 3
        
        # ★ v6: chattering模式的音频不再给food加分
        # 低centroid+低flatness+高hf_ratio是chattering特征，不是food
        if centroid < 800 and hf_ratio > 0.25 and flatness < 0.02:
            scores["food"] -= 5  # 扣food分，这是chattering
        
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
        
        if hf_ratio > 0.30 and flatness > 0.02:  # ★ v6: 只给非chattering的高hf_ratio加分
            scores["food"] += 4
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
        
        # ★ v6.1新增: 低centroid + 高f0_std + 不是wailing = 烦躁/不满
        # 反复叫但不是痛苦哀鸣(f0逐帧稳定，只是段间音高跨度大)
        # ★ v6.2: 排除反复meow(高f0_std>150+低fdm<10)——那是food不是烦躁
        if centroid < 1500 and f0_std > 40 and f0_frame_diff_median < 15 and not (f0_std > 150 and f0_frame_diff_median < 10):
            scores["angry"] += 12  # 烦躁反复叫的强信号（v6.1: 8→12）
        
        # ★ v6.2新增: 连续抱怨 — 高voiced_ratio(连续发声) + 低centroid + 高f0_std
        # voiced_ratio>0.6 = 几乎不停地在叫 = 不满/抱怨，不是food那种断续meow
        # 但排除反复meow(高f0_std+低fdm)——那是food不是抱怨
        if centroid < 1500 and 40 < f0_std <= 150 and voiced_ratio > 0.6 and f0_frame_diff_median < 15:
            scores["angry"] += 8  # 连续抱怨额外加分
        
        # --- PAIN (评分兜底) ---
        # ★ v6.1: 不是真正wailing时pain降权
        if duration > 2.5:
            scores["pain"] += 5
        elif duration > 1.5:
            scores["pain"] += 3
        
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
        
        # ★ v6.1: f0_frame_diff_median低说明是反复meow而非持续哀鸣，pain扣分
        if f0_frame_diff_median < 10 and f0_std > 30:
            scores["pain"] -= 6  # 不是wailing但f0_std高=多段meow，不是pain
        
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
