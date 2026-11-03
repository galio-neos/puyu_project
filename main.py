"""
================================================================================
主模块 — 无地图主动目标搜索
================================================================================

【本文件的作用】
  这是整个项目的核心文件。包含：
    1. 感知模块  — 模拟检测器 / 显著性检测器
    2. 搜索策略  — 4 个 baseline（基线）+ 4 个创新 + 1 个组合 = 9 个搜索策略
    3. 实验框架  — 批量运行、收集指标、打印结果、消融实验（ablation study）

【数据流（从上到下）】
  ExperimentRunner.run()
    → 生成合成场景 (SyntheticSceneGenerator)
    → 生成候选窗口 (WindowGrid)
    → 对每个搜索策略 (Searcher):
        → 每步选择一个窗口
        → 观察窗口 (_observe)
        → 获取检测置信度 (SimulatedDetector)
        → 决定下一步看哪里
        → 置信度 >= 0.85 → 停止
    → 收集指标 (MetricsTracker)
    → 打印结果表格

【关键设计原则（初学者必读）】
  Q: 为什么显著性可以预计算，但检测置信度不行？
  A: 显著性是一种"低成本底层特征"，对任何图像都能快速算出（类似人眼
     扫一眼就知道"那里很花/很亮/很突兀"）。检测置信度需要实际运行检测器、
     分析图像内容——在仿真中，我们把它建模为"观察后才获得的 feedback（反馈）"。
     这样设计才能真正比较不同策略在"逐步获取信息"场景下的表现。

  Q: 未观察的窗口怎么估算置信度？
  A: _estimate_conf() 用空间插值（spatial interpolation）：已观察窗口的置信度 × 距离权重。
     离已观察的高置信度窗口越近 → 估算值越高。
     这模拟了"目标一般不会突然消失——附近区域也值得看看"的直觉。

================================================================================

文件结构速览:
  行 70:    SimulatedDetector      模拟目标检测器
  行 100:   SaliencyDetector       显著性检测器
  行 140:   TrialResult / MetricsTracker  指标数据结构
  行 180:   BaseSearcher           抽象基类（所有策略的父类）
  行 280:   B1_SequentialScan      顺序扫描
  行 305:   B2_RandomScan          随机扫描
  行 350:   B3_SaliencyGuided      显著性引导
  行 390:   B4_DetectionConfidence 检测置信度搜索
  行 450:   I1_InhibitionOfReturn  返回抑制 ⭐
  行 540:   I2_ScenePriorGuided    场景先验 ⭐
  行 610:   I3_CoarseToFine        粗到细 ⭐
  行 770:   I4_VisualMemory        视觉记忆 ⭐
  行 870:   CombinedSearcher       组合策略
  行 1000:  ExperimentRunner       实验运行器
  行 1140:  run_ablation_study     消融实验
  行 1205:  main()                 入口
================================================================================
"""

import time
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import warnings
import zlib

import numpy as np
import cv2

from config import (
    config, ExperimentConfig, WindowConfig, DetectionConfig,
    SaliencyConfig, ScenePriorConfig, MemoryConfig,
)
from environment import (
    WindowGrid, ObservationWindow, ObservationResult,
    PanoramaScene, SyntheticSceneGenerator,
)

# 忽略 numpy 的一些运行时警告（如除零、空切片等）
warnings.filterwarnings("ignore")

# 修复 Windows 中文终端 GBK 编码下的 Unicode 输出问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _category_seed(category: str) -> int:
    """
    把目标类别名映射成稳定的整数偏移（用于构造 trial 随机种子）。

    【为什么不用内置 hash()？】
      Python 的字符串 hash() 每个进程都有一个随机盐（PYTHONHASHSEED），
      同一个字符串在不同进程里 hash 值不同。若用它构造随机种子，会导致
      每次运行 main.py 得到的实验数字都不一样——无法复现。
      这里改用 zlib.crc32（跨进程稳定），保证同一类别永远得到同一个偏移。
    """
    return zlib.crc32(category.encode("utf-8")) % 1000


# ============================================================================
# 一、模拟目标检测器
# ============================================================================

class SimulatedDetector:
    """
    模拟目标检测器 —— 用数学函数替代真实的 YOLO（You Only Look Once）/
    DETR（Detection Transformer）等检测模型。

    【为什么用模拟？】
      1. 不需要 GPU（Graphics Processing Unit，图形处理器），
         不需要下载几百 MB 的模型权重（weights）
      2. 检测结果是确定性的（deterministic，给定种子 seed），方便对比策略差异
      3. 可以精确控制噪声水平（noise level），做消融实验（ablation study）
      4. 搜索策略代码不依赖具体检测器，换了真实模型照样用

    【核心公式】
      confidence = 0.95 * exp( -distance² / (2 * 80²) ) + 随机噪声（random noise）

      直观理解：
        - 窗口离目标 0px   → confidence ≈ 0.95（最高）
        - 窗口离目标 80px  → confidence ≈ 0.95 * 0.606 ≈ 0.58
        - 窗口离目标 200px → confidence ≈ 0.95 * 0.044 ≈ 0.04（几乎检测不到）

      如果窗口确实包含目标（has_target_at），保底 confidence ≥ 0.95 * 0.8 = 0.76。
      这确保目标不会被噪声完全掩盖。
    """

    def __init__(self, det_config: DetectionConfig = None):
        """det_config 为 None 时使用默认配置"""
        self.cfg = det_config or DetectionConfig()

    def detect(self, window: ObservationWindow, scene: PanoramaScene) -> Tuple[float, bool]:
        """
        对给定窗口执行"检测"，返回 (置信度 confidence, 是否找到)。

        参数:
          window: 当前观察的窗口
          scene: 全景场景（用于计算距离到目标）

        返回:
          confidence: [0, 1] 的浮点数
          found: confidence >= 0.70 则为 True
        """

        # 1. 计算窗口中心到最近目标的像素距离
        dist = scene.distance_to_nearest_target(window)

        # 2. 高斯衰减（Gaussian decay）：距离越远，置信度越低
        #    exp(-0.5 * (d/σ)²) = 高斯函数，σ（sigma）=80 控制衰减速度
        confidence = self.cfg.sim_peak_confidence * np.exp(
            -0.5 * (dist / self.cfg.sim_decay_sigma) ** 2
        )

        # 3. 添加随机噪声（noise，模拟真实检测器的不确定性）
        #    np.random.normal(0, 0.15)：均值为 0，标准差（std）0.15 的正态分布（normal distribution）
        noise = np.random.normal(0, self.cfg.sim_noise_std)
        confidence = np.clip(confidence + noise, 0.0, 1.0)  # clamp（裁剪）到 [0, 1]

        # 4. 保底逻辑：如果目标确实在窗口内，确保置信度不会太低
        if scene.has_target_at(window):
            confidence = max(confidence, self.cfg.sim_peak_confidence * 0.8)  # ≥ 0.76

        # 5. 判断：置信度超过 found_threshold（检测阈值，0.70）则算"找到"
        found = confidence >= self.cfg.found_threshold
        return float(confidence), found


# ============================================================================
# 二、显著性检测器
# ============================================================================

class SaliencyDetector:
    """
    基于 Spectral Residual（SR，谱残差法）的显著性检测（Saliency Detection）。

    【什么是显著性（Saliency）？】
      图像中"一眼就能注意到"的区域。比如白墙上的黑点、草地上的红花。
      显著性高的地方 → 可能有物体 → 值得优先观察。

    【Spectral Residual（谱残差法）原理（5 行版）】
      1. FFT（Fast Fourier Transform，快速傅里叶变换）
         → 把图像从"空间域"（spatial domain，像素坐标）变换到
           "频率域"（frequency domain，频谱 spectrum）
      2. log → 取对数幅度谱（log magnitude spectrum）
      3. blur（均值滤波 / mean blur）
         → 得到"平均频谱"（代表图像的"常见模式"）
      4. 相减 → 原频谱 - 平均频谱 = "异常模式" = 显著区域
      5. IFFT（Inverse FFT，逆快速傅里叶变换）
         → 逆变换回空间域，得到显著性图（saliency map）

    【为什么可以预计算（precompute）？】
      显著性只需要窗口图像本身就能算出，不需要知道目标在哪。
      像"这个区域颜色突兀"这类信息是"免费的"——在任何搜索步骤前都能算好。
    """

    def __init__(self, sal_config: SaliencyConfig = None):
        self.cfg = sal_config or SaliencyConfig()

    def compute(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算图像的显著性分数（saliency score）和显著性图（saliency map）。

        参数:
          image: RGB 图像 (H, W, 3)

        返回:
          score: 平均显著性 [0, 1]
          saliency_map: 像素级显著性图 (H, W)
        """

        # 1. 转灰度图（grayscale，显著性通常不需要颜色信息）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        # 2. 缩放到 FFT 尺寸（64×64），大幅加速计算
        gray_resized = cv2.resize(gray, self.cfg.sr_fft_size)

        # 3. 二维傅里叶变换（2D FFT）→ 得到幅度谱（magnitude spectrum）和相位谱（phase spectrum）
        fft = np.fft.fft2(gray_resized)                   # 复数矩阵（complex matrix）
        fft_shifted = np.fft.fftshift(fft)                # 低频移到中心（shift）
        log_magnitude = np.log(np.abs(fft_shifted) + 1e-8) # 对数幅度谱（+1e-8 防止 log(0)）
        phase = np.angle(fft_shifted)                      # 相位谱（保留结构信息）

        # 4. 均值滤波（mean blur）→ 得到"平均频谱" → 相减得到"谱残差（spectral residual）"
        avg_log_mag = cv2.blur(log_magnitude,
                               (self.cfg.sr_blur_size, self.cfg.sr_blur_size))
        spectral_residual = log_magnitude - avg_log_mag    # 异常 = 原 - 平均

        # 5. 逆傅里叶变换（IFFT）：谱残差 + 原始相位 → 重建显著性图
        combined = np.exp(spectral_residual + 1j * phase)  # 1j = 虚数单位（imaginary unit）√(-1)
        saliency = np.abs(np.fft.ifft2(np.fft.ifftshift(combined))) ** 2

        # 6. GaussianBlur（高斯平滑）+ resize 回原始尺寸
        saliency = cv2.GaussianBlur(saliency,
                                    (self.cfg.sr_blur_size, self.cfg.sr_blur_size), 2.0)
        saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]))

        # 7. 归一化（normalize）到 [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        score = float(np.mean(saliency))  # 平均显著性作为该窗口的"显著分数"
        return score, saliency


# ============================================================================
# 三、指标数据容器
# ============================================================================

@dataclass
class TrialResult:
    """
    存储单次搜索试验的全部数据。

    【一次 trial（试验）是什么？】
      一次 trial = 一种策略在一个场景上搜索一种目标的完整过程。
      例如："B1 顺序扫描在场景#3上找水杯，用了 35 步找到"

    【各字段含义】
      - strategy: 策略名（如 "B1_SequentialScan"）
      - target_category: 找什么（如 "水杯"）
      - found_target: 最终找到了吗？
      - false_stop: 是否误停（false stop）？（置信度达到阈值但实际没找到）
      - total_steps: 总共观察了多少步
      - repeat_count: 重复观察了多少次
      - observations: 每一步的详细结果列表
      - confidences: 每一步的置信度列表（用于画曲线 plot）
    """
    strategy: str = ""
    target_category: str = ""
    trial_id: int = 0                          # 全局唯一编号
    found_target: bool = False                 # 在预算（budget）内找到目标？
    false_stop: bool = False                   # 声称找到但实际没有？（false positive）
    total_steps: int = 0                       # 使用了多少步
    repeat_count: int = 0                      # 重复观察次数
    observations: List = field(default_factory=list)    # 每步的 ObservationResult
    confidences: List[float] = field(default_factory=list)  # 每步的置信度


class MetricsTracker:
    """
    汇总所有 trial 的结果，计算四种核心指标（metrics）。

    【四种指标】
      1. 发现率 (discovery_rate) = 成功 trial 数 / 总 trial 数
         → 衡量策略"能不能找到目标"

      2. 平均步数 (avg_steps) = 成功 trial 的平均观察步数
         → 衡量策略"找得多快"

      3. 重复观察率 (repeat_rate) = 重复观察次数 / 总观察次数
         → 衡量策略"是否浪费步数"

      4. 错误停止率 (false_stop_rate) = 误停 trial 数 / 总 trial 数
         → 衡量策略"是否把错误当目标"（false positive rate）
    """

    def __init__(self):
        self.results: List[TrialResult] = []

    def add(self, result: TrialResult):
        """添加一次试验结果"""
        self.results.append(result)

    def _repeat_rate(self, result: TrialResult) -> float:
        """
        基于像素重叠计算单次 trial 的"重复观察率"。

        【为什么不用 repeat_count？】
          原来的 repeat_count 只统计"完全相同的窗口被看两次"，但所有策略
          都刻意不会重复选同一个窗口，所以它恒为 0（是死代码）。
          真正浪费步数的是"重叠观察"——相邻窗口 stride=100、大小 200，
          有 50% 面积重叠。所以改成统计"新观察的窗口里有多少像素是之前已经看过的"。

        【返回值】
          平均每步的"已看过像素占比"，范围 [0, 1]。
          值越大 → 策略越喜欢反复看重叠区域（浪费步数）。
        """
        if not result.observations:
            return 0.0

        # 推断全景尺寸：取所有窗口右下角坐标的极大值
        h = max((o.window.y + o.window.height) for o in result.observations)
        w = max((o.window.x + o.window.width) for o in result.observations)

        mask = np.zeros((h, w), dtype=bool)   # 像素级"已观察"标记
        overlap_sum = 0.0

        for o in result.observations:         # 按观察顺序累加
            win = o.window
            x1, y1 = max(0, win.x), max(0, win.y)
            x2, y2 = min(w, win.x + win.width), min(h, win.y + win.height)
            if x2 <= x1 or y2 <= y1:
                continue
            # 这一步"新窗口里已经被看过的像素占比"
            overlap_sum += float(np.mean(mask[y1:y2, x1:x2]))
            # 把这个窗口标记为"已观察"
            mask[y1:y2, x1:x2] = True

        return overlap_sum / len(result.observations)

    def summary(self) -> Dict:
        """汇总所有试验，返回指标字典（dict）"""
        if not self.results:
            return {}

        total = len(self.results)

        # ----- 发现率：多少比例找到目标 -----
        found = sum(1 for r in self.results if r.found_target)

        # ----- 错误停止率：多少比例误判（false positive）-----
        false_stops = sum(1 for r in self.results if r.false_stop)

        # ----- 平均步数：只算成功找到的那些 trial -----
        found_steps = [r.total_steps for r in self.results if r.found_target]

        # ----- 重复观察率：每个 trial 的重复比例再取平均 -----
        all_repeats = [self._repeat_rate(r) for r in self.results]

        return {
            "trials": total,
            "discovery_rate": found / total,
            "avg_steps": np.mean(found_steps) if found_steps else float("nan"),
            "median_steps": np.median(found_steps) if found_steps else float("nan"),
            "repeat_rate": np.mean(all_repeats),
            "false_stop_rate": false_stops / total,
        }


# ============================================================================
# 四、抽象基类 — 所有搜索策略的"模板"
# ============================================================================

class BaseSearcher(ABC):
    """
    所有搜索策略的父类。定义了公共接口（interface）和工具方法。

    【继承关系（inheritance）】
      BaseSearcher（ABC，Abstract Base Class，抽象基类）
        ├── B1_SequentialScan
        ├── B2_RandomScan
        ├── B3_SaliencyGuided
        ├── B4_DetectionConfidence
        ├── I1_InhibitionOfReturn
        ├── I2_ScenePriorGuided
        ├── I3_CoarseToFine
        ├── I4_VisualMemory
        └── Combined

    【子类需要做什么】
      只需实现 search(scene, grid) → TrialResult 这一个方法。
      基类已经把"观察"、"估算"、"预计算"等公共操作封装（encapsulate）好了。

    【缓存设计（cache）】
      四种缓存字典，key 都是窗口的 idx（唯一编号）：
        _sal_cache[idx]      → 显著性分数（预计算，一次算好反复用）
        _scene_cache[idx]    → 场景分布（预计算）
        _observed_conf[idx]  → 观察到的置信度（观察后才填入）
        _window_centers[idx] → 窗口中心坐标（预计算）
    """

    def __init__(self, name: str, max_steps: int = 50):
        """
        参数:
          name: 策略名称（用于打印和排名 ranking）
          max_steps: 搜索预算（budget，最多观察几步）
        """
        self.name = name
        self.max_steps = max_steps

        # 两个"传感器"（sensor）
        self.detector = SimulatedDetector()     # 模拟目标检测器
        self.saliency_det = SaliencyDetector()  # 显著性检测器

        # ---- 四个缓存字典（cache）----
        # 显著性缓存：窗口 → 显著性分数
        self._sal_cache: Dict[int, float] = {}
        # 场景分布缓存：窗口 → {"桌子": 0.6, "地板": 0.3, ...}
        self._scene_cache: Dict[int, Dict[str, float]] = {}
        # 已观察置信度：窗口 → 检测置信度（只包含已经观察过的窗口）
        self._observed_conf: Dict[int, float] = {}
        # 窗口中心缓存：窗口 → (cx, cy)
        self._window_centers: Dict[int, Tuple[float, float]] = {}

    @abstractmethod
    def search(self, scene: PanoramaScene, grid: WindowGrid) -> TrialResult:
        """
        子类必须实现（implement）这个方法（由 @abstractmethod 装饰器强制要求）。

        【这是整个项目最核心的接口（interface）】
          输入: 全景场景 + 候选窗口网格
          输出: 一次搜索试验的完整结果

        每个搜索策略的不同之处，就在于这个方法里的"选择下一个窗口"的逻辑。
        """
        ...

    def _precompute_with_scene(self, windows: List[ObservationWindow],
                                scene: PanoramaScene):
        """
        预计算所有窗口的"免费信息"。

        【什么是"免费信息"？】
          显著性 + 场景分布 — 不需要实际观察就能获得。
          像人眼扫视一样，瞟一眼就知道"哪里花哨""哪里是桌子"。

        【何时调用】
          每个 search() 方法的开头调用一次，把所有窗口的信息缓存（cache）起来，
          后续步骤直接从缓存取，避免重复计算。
        """
        self._sal_cache.clear()
        self._scene_cache.clear()
        self._window_centers.clear()
        self._observed_conf.clear()     # 新搜索 → 清空之前观察的记忆

        for w in windows:
            # 裁剪窗口图像 → 计算显著性
            crop = w.crop_from(scene.image)
            sal_score, _ = self.saliency_det.compute(crop)
            self._sal_cache[w.idx] = sal_score

            # 获取场景分布（scene distribution）
            self._scene_cache[w.idx] = scene.get_scene_distribution(w)

            # 记录窗口中心坐标
            self._window_centers[w.idx] = w.center

    # ---- 便捷访问器（accessor，从缓存取值） ----
    def _sal(self, idx: int) -> float:
        return self._sal_cache.get(idx, 0.0)

    def _scene(self, idx: int) -> Dict[str, float]:
        return self._scene_cache.get(idx, {})

    def _center(self, idx: int) -> Tuple[float, float]:
        return self._window_centers.get(idx, (0.0, 0.0))

    def _observe(self, window: ObservationWindow, scene: PanoramaScene,
                 step: int) -> ObservationResult:
        """
        执行一次观察：看向这个窗口，获取检测结果。

        【关键】这里才真正调用 detect()观察前不知道置信度。

        返回的 ObservationResult 包含：
          - 看到什么图像
          - 检测置信度多少
          - 显著性多少
          - 是什么场景
          - 是否找到目标
        """
        crop = window.crop_from(scene.image)

        # ⭐ 核心操作：实际运行检测器，获取置信度
        conf, found = self.detector.detect(window, scene)

        # 记录到"已观察"缓存（后续 _estimate_conf 会用到）
        self._observed_conf[window.idx] = conf

        # 从缓存读取预计算的免费信息
        sal_score = self._sal(window.idx)
        scene_dist = self._scene(window.idx)

        return ObservationResult(
            window=window, image=crop,
            detection_confidence=conf, saliency_score=sal_score,
            saliency_map=None, scene_labels=scene_dist,
            found_target=found, step=step,
        )

    def _estimate_conf(self, window_idx: int, sigma: float = 200.0) -> float:
        """
        【关键方法】对未观察窗口的检测置信度进行空间插值（spatial interpolation）估算。

        这是 B4、I1、I4、Combined 四种策略的共有基础。

        【人类直觉】
          如果你已经观察了窗口 A（置信度 0.8），还没看过窗口 B。
          那么 B 的"估算置信度"取决于：
            - B 离 A 有多近？（越近 → 估算越高，目标不太可能突然消失）
            - A 的置信度多高？（越高 → B 附近也越可能有目标）

        【数学】
          estimate(B) = Σ(已观察窗口_i 的置信度 × 权重_i) / Σ(权重_i)
          其中 权重_i = exp(-0.5 × (B到i的距离 / 200)²)

          距离 200px → 权重（weight）≈ 0.61
          距离 400px → 权重 ≈ 0.14
          距离 600px → 权重 ≈ 0.01

        【如果没有已观察窗口】
          返回 0.0 — 完全不知道，没有任何先验信息（prior information）。
        """
        if not self._observed_conf:
            return 0.0

        cx, cy = self._center(window_idx)     # 目标窗口的中心坐标
        total_weight = 0.0
        weighted_sum = 0.0

        for obs_idx, conf in self._observed_conf.items():
            ox, oy = self._center(obs_idx)    # 已观察窗口的中心坐标

            # 两个窗口中心的欧几里得距离（Euclidean distance）
            dist = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)

            # 高斯权重（Gaussian weight）：距离越近权重越大
            weight = np.exp(-0.5 * (dist / sigma) ** 2)

            weighted_sum += conf * weight     # 置信度 × 权重
            total_weight += weight            # 权重累加

        # 加权平均（weighted average）
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _stop_thresh(self) -> float:
        """停止搜索的置信度阈值（stop threshold，默认 0.85）

        合成实验用默认 0.85；真实图（YOLO 置信度偏低）可通过
        `searcher._stop_threshold = 0.6` 覆盖，无需改任何策略代码。
        """
        return getattr(self, "_stop_threshold", DetectionConfig().stop_threshold)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                   五、BASELINE（基线）搜索策略（4 个）                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝


# ============================================================================
# B1: 顺序扫描 — 最朴素的策略
# ============================================================================

class SequentialScanSearcher(BaseSearcher):
    """
    B1 — 顺序扫描（Sequential Scan）

    【策略】
      从左到右、从上到下，按 WindowGrid 生成的顺序依次观察。
      完全不做任何智能决策——这是所有策略的"地板"（性能下限 lower bound）。

    【优点】
      - 最简单、最容易实现
      - 保证覆盖所有区域（只是可能太慢）

    【缺点】
      - 目标在右下角的话要等到最后几步才看到
      - 对目标位置没有任何"猜测"
    """

    def __init__(self, max_steps: int = 50):
        super().__init__("B1_SequentialScan", max_steps)

    def search(self, scene: PanoramaScene, grid: WindowGrid) -> TrialResult:
        # 1. 生成标准尺度的候选窗口列表
        windows = grid.generate(1.0)

        # 2. 预计算所有窗口的"免费信息"（显著性、场景分布）
        self._precompute_with_scene(windows, scene)

        # 3. 初始化试验结果记录
        result = TrialResult(strategy=self.name, target_category=scene.target_category)
        observed_idx: Set[int] = set()       # 已观察窗口编号集合（set，用于去重）
        stop_thresh = self._stop_thresh()    # 0.85

        # 4. 逐步观察
        for step in range(min(self.max_steps, len(windows))):
            # B1 的核心：直接按列表顺序取下一个窗口
            window = windows[step]

            # 检查是否重复观察（顺序扫描理论上不会重复，但保留检查）
            if window.idx in observed_idx:
                result.repeat_count += 1
            observed_idx.add(window.idx)

            # 执行观察
            obs = self._observe(window, scene, step)
            result.observations.append(obs)
            result.confidences.append(obs.detection_confidence)

            # 检查停止条件
            if obs.detection_confidence >= stop_thresh:
                result.found_target = obs.found_target
                result.false_stop = not scene.has_target_at(obs.window)
                result.total_steps = step + 1
                return result

        # 预算（budget）用完，未找到
        result.total_steps = min(self.max_steps, len(windows))
        return result


# ============================================================================
# B2: 随机扫描
# ============================================================================

class RandomScanSearcher(BaseSearcher):
    """
    B2 — 随机扫描（Random Scan）

    【策略】
      从未观察的窗口中随机选一个。比顺序扫描好的原因是：
      不会因为目标在"最后几个"而系统性错过——每个窗口都有被提前选中的概率。

    【和 B1 的对比】
      B1: 如果目标在第 70 个窗口 → 必须等 70 步
      B2: 如果目标在第 70 个窗口 → 每步有 1/75 概率选中 → 期望约 38 步
    """

    def __init__(self, max_steps: int = 50):
        super().__init__("B2_RandomScan", max_steps)
        # 注意：这里不再创建独立的 RandomState()，而是直接用全局 np.random。
        # 因为 ExperimentRunner 在每次 trial 前会调用 np.random.seed(trial_seed)，
        # 用全局随机数生成器才能让"随机扫描"也随 trial 种子复现；
        # 否则 RandomState() 无参数时会从系统熵随机，每次运行结果都不一样。

    def search(self, scene: PanoramaScene, grid: WindowGrid) -> TrialResult:
        windows = grid.generate(1.0)
        self._precompute_with_scene(windows, scene)
        result = TrialResult(strategy=self.name, target_category=scene.target_category)
        observed_idx: Set[int] = set()
        available = list(range(len(windows)))  # 还未被选过的窗口索引（available candidates）
        stop_thresh = self._stop_thresh()

        for step in range(min(self.max_steps, len(windows))):
            # B2 的核心：从未观察列表中随机弹出一个（pop）
            idx = np.random.randint(0, len(available))  # 随机位置（用全局 RNG，随 trial 种子复现）
            wi = available.pop(idx)                     # 弹出（自动从列表中移除）
            window = windows[wi]

            if window.idx in observed_idx:
                result.repeat_count += 1
            observed_idx.add(window.idx)

            obs = self._observe(window, scene, step)
            result.observations.append(obs)
            result.confidences.append(obs.detection_confidence)

            if obs.detection_confidence >= stop_thresh:
                result.found_target = obs.found_target
                result.false_stop = not scene.has_target_at(obs.window)
                result.total_steps = step + 1
                return result

        result.total_steps = min(self.max_steps, len(windows))
        return result


# ============================================================================
# B3: 显著性引导搜索
# ============================================================================

class SaliencyGuidedSearcher(BaseSearcher):
    """
    B3 — 显著性引导搜索（Saliency-Guided Search）

    【策略】
      在所有未观察窗口中，优先选显著性最高的那个。
      显著性高低代表"这个地方看起来有没有东西"。

    【为什么效果可能不好？】
      显著性 = 视觉突出程度 ≠ 目标存在概率。
      比如"蓝天"显著性低但可能有鸟，"花丛"显著性高但可能没你要找的杯子。
      显著性是一个粗粒度的"注意力引导"，不能替代目标检测。

    【实验结果】
      在合成场景中表现最差（9.0%），因为：
      - 合成场景是均匀色块（uniform color blocks），显著性差异不大
      - 目标（小圆点）本身不是场景中最显著的东西
      - 在真实图像中应有更好表现
    """

    def __init__(self, max_steps: int = 50):
        super().__init__("B3_SaliencyGuided", max_steps)

    def search(self, scene: PanoramaScene, grid: WindowGrid) -> TrialResult:
        windows = grid.generate(1.0)
        self._precompute_with_scene(windows, scene)
        result = TrialResult(strategy=self.name, target_category=scene.target_category)
        observed_mask = np.zeros(len(windows), dtype=bool)  # True = 已观察（boolean mask）
        stop_thresh = self._stop_thresh()

        for step in range(min(self.max_steps, len(windows))):
            # B3 的核心：从未观察窗口中选出显著性最高的
            available = [(i, self._sal(windows[i].idx))
                         for i in range(len(windows)) if not observed_mask[i]]
            if not available:
                break

            # max(..., key=lambda x: x[1]) → 按显著性排序，取最高
            # available 里每个元素是元组 (编号, 显著性)，不能直接比较。
            # key=lambda x: x[1] 表示"比较时用每个元素的第 2 项（显著性）来比"，
            # 最后 [0] 取出分数最高的那个窗口的编号。
            best_i = max(available, key=lambda x: x[1])[0]
            window = windows[best_i]
            observed_mask[best_i] = True

            obs = self._observe(window, scene, step)
            result.observations.append(obs)
            result.confidences.append(obs.detection_confidence)

            if obs.detection_confidence >= stop_thresh:
                result.found_target = obs.found_target
                result.false_stop = not scene.has_target_at(obs.window)
                result.total_steps = step + 1
                return result

        result.total_steps = min(self.max_steps, len(windows))
        return result


# ============================================================================
# B4: 检测置信度搜索 — 第一个"智能"策略
# ============================================================================

class DetectionConfidenceSearcher(BaseSearcher):
    """
    B4 — 检测置信度搜索（Detection Confidence Search，置信度梯度上升 gradient ascent）

    【策略】
      1. 先从最高显著性窗口开始观察
      2. 每次观察后，用 _estimate_conf() 估算所有未观察窗口的置信度
      3. 选择估算置信度最高的窗口作为下一步目标

    【为什么比 B1/B2/B3 好？】
      因为它利用"已观察窗口的反馈（feedback）"来指导下一步——
      如果某个区域检测到高置信度，就往那个方向继续探索。
      这比"盲扫"（B1）或"随机"（B2）或"只看显著性"（B3）更聪明。

    【和 I1/I2/I4 的关系】
      B4 是"智能搜索"的起点模板（template）——
      I1 在 B4 基础上加了"返回抑制（IOR）"，
      I2 加了"场景先验（Scene Prior）"，
      I4 加了"视觉记忆（Visual Memory）"，
      Combined 加了全部。
    """

    def __init__(self, max_steps: int = 50):
        super().__init__("B4_DetectionConfidence", max_steps)

    def search(self, scene: PanoramaScene, grid: WindowGrid) -> TrialResult:
        windows = grid.generate(1.0)
        self._precompute_with_scene(windows, scene)
        result = TrialResult(strategy=self.name, target_category=scene.target_category)
        observed_idx: Set[int] = set()
        stop_thresh = self._stop_thresh()

        # ---- 第 0 步：从显著性最高的窗口开始 ----
        all_sal = [(i, self._sal(w.idx)) for i, w in enumerate(windows)]
        start_i = max(all_sal, key=lambda x: x[1])[0]

        window = windows[start_i]
        observed_idx.add(window.idx)
        obs = self._observe(window, scene, 0)
        result.observations.append(obs)
        result.confidences.append(obs.detection_confidence)

        # 第一步就找到了？
        if obs.detection_confidence >= stop_thresh:
            result.found_target = obs.found_target
            result.false_stop = not scene.has_target_at(obs.window)
            result.total_steps = 1
            return result

        # ---- 第 1~N 步：基于空间插值（spatial interpolation）选择下一个窗口 ----
        for step in range(1, min(self.max_steps, len(windows))):
            best_w, best_s = None, -1.0

            for w in windows:
                if w.idx in observed_idx:
                    continue  # 跳过已观察的

                # ⭐ B4 的核心评分公式
                # 已观察窗口的空间插值估算 + 轻微显著性偏置（bias）
                est_conf = self._estimate_conf(w.idx)
                score = est_conf + 0.1 * self._sal(w.idx)
                #       ↑ 主要依据            ↑ 辅助：如果没有任何观察信息，显著性提供初始方向

                if score > best_s:
                    best_s, best_w = score, w

            if best_w is None:
                break

            window = best_w
            if window.idx in observed_idx:
                result.repeat_count += 1
            observed_idx.add(window.idx)

            obs = self._observe(window, scene, step)
            result.observations.append(obs)
            result.confidences.append(obs.detection_confidence)

            if obs.detection_confidence >= stop_thresh:
                result.found_target = obs.found_target
                result.false_stop = not scene.has_target_at(obs.window)
                result.total_steps = step + 1
                return result

        result.total_steps = min(self.max_steps, len(windows))
        return result


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                   六、创新搜索策略（4 个）⭐                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝


# ============================================================================
# I1: 返回抑制（Inhibition of Return） — 效果最强的单一创新 🥇
# ============================================================================

class InhibitionOfReturnSearcher(BaseSearcher):
    """
    I1 — 返回抑制（IOR, Inhibition of Return）

    【为什么需要 IOR？】
      没有 IOR 时，如果检测器对某个区域给出高置信度（即使是假阳性 false positive），
      搜索就会"卡"在那个区域附近反复看，浪费步数。
      IOR 强制搜索去探索新区域。

    【灵感来源】
      人类视觉系统中的"返回抑制"机制（Posner & Cohen, 1984）——
      刚注意过的位置在约 300ms~3s 内被抑制，视线不会重复扫过同一位置。

    【实现】
      1. 维护一个像素级"观察计数图" observed_count
         （每个像素被看过几次）
      2. 选择下一个窗口时，已观察比例越高的窗口得分越低：
         score = est_conf × (1 - observed_ratio × 0.6)
                         + 0.1 × saliency × (1 - observed_ratio)

    【参数】
      ior_strength（IOR 强度）= 0.6：IOR 强度。0 = 无抑制，1 = 完全禁止回看。

    【实验结果】
      发现率 70.7% — 所有单一策略中最高，比 B4 (50.0%) 提升 +20.7pp（百分点）。
    """

    def __init__(self, max_steps: int = 50, ior_strength: float = 0.6):
        super().__init__("I1_InhibitionOfReturn", max_steps)
        self.ior_strength = ior_strength  # IOR 强度

    def search(self, scene: PanoramaScene, grid: WindowGrid) -> TrialResult:
        windows = grid.generate(1.0)
        self._precompute_with_scene(windows, scene)
        result = TrialResult(strategy=self.name, target_category=scene.target_category)
        observed_idx: Set[int] = set()

        # ⭐ IOR 的核心数据结构：像素级观察计数
        # 形状 (600, 1600)，每个像素记录被观察过的次数
        observed_count = np.zeros((scene.height, scene.width), dtype=np.float32)
        stop_thresh = self._stop_thresh()

        # ---- 第 0 步 ----
        # 遍历所有窗口，把每个窗口打包成 (编号, 显著性分数) 的元组存进列表
        all_sal = [(i, self._sal(w.idx)) for i, w in enumerate(windows)]
        start_i = max(all_sal, key=lambda x: x[1])[0]
        window = windows[start_i]
        observed_idx.add(window.idx)

        # 更新观察计数：把这个窗口覆盖的像素 +1
        x1, y1, x2, y2 = window.bounds
        y1c, y2c = max(0, y1), min(scene.height, y2)
        x1c, x2c = max(0, x1), min(scene.width, x2)
        observed_count[y1c:y2c, x1c:x2c] += 1.0
        #一开始全是0，每看过一次就加一

        obs = self._observe(window, scene, 0)
        result.observations.append(obs)
        result.confidences.append(obs.detection_confidence)

        if obs.detection_confidence >= stop_thresh:
            result.found_target = obs.found_target
            result.false_stop = not scene.has_target_at(obs.window)
            result.total_steps = 1
            return result

        # ---- 第 1~N 步 ----
        for step in range(1, min(self.max_steps, len(windows))):
            best_w, best_s = None, -1.0

            for w in windows:
                if w.idx in observed_idx:
                    continue

                # 边界检查（bounds check）
                bx1, by1, bx2, by2 = w.bounds
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(scene.width, bx2), min(scene.height, by2)
                if bx2 <= bx1 or by2 <= by1:
                    continue

                # ⭐ IOR 核心计算
                # 1. 这个窗口有多少像素已经被看过了？
                patch = observed_count[by1:by2, bx1:bx2]
                observed_ratio = np.mean(patch > 0)  # [0, 1]

                # 2. 估算置信度
                est_conf = self._estimate_conf(w.idx)

                # 3. IOR 评分公式（核心：抑制已看区域，逼出对新区域的覆盖）
                #    置信度 × (1 - 已观察比例 × IOR强度)
                #    + 显著性 × (1 - 已观察比例)
                score = (est_conf * (1.0 - observed_ratio * self.ior_strength) +
                         0.1 * self._sal(w.idx) * (1.0 - observed_ratio))

                if score > best_s:
                    best_s, best_w = score, w

            if best_w is None:
                break

            window = best_w
            if window.idx in observed_idx:
                result.repeat_count += 1
            observed_idx.add(window.idx)

            # 更新观察计数
            bx1, by1, bx2, by2 = window.bounds
            observed_count[max(0, by1):min(scene.height, by2),
                           max(0, bx1):min(scene.width, bx2)] += 1.0

            obs = self._observe(window, scene, step)
            result.observations.append(obs)
            result.confidences.append(obs.detection_confidence)

            if obs.detection_confidence >= stop_thresh:
                result.found_target = obs.found_target
                result.false_stop = not scene.has_target_at(obs.window)
                result.total_steps = step + 1
                return result

        result.total_steps = min(self.max_steps, len(windows))
        return result


# ============================================================================
# I2: 场景先验引导搜索 — 消融实验中最关键的组件
# ============================================================================

class ScenePriorGuidedSearcher(BaseSearcher):
    """
    I2 — 场景先验引导搜索（Scene Prior Guided Search）

    【为什么需要场景先验（Scene Prior）？】
      显著性告诉你"哪里显眼"，但不能告诉你"杯子更可能在桌子上而不是天上"。
      场景先验把"常识"编码进去：优先搜索目标可能出现的地方。

    【实现】
      1. 从 config 中加载目标-场景共现表（co-occurrence table）
         （如 水杯→桌子:0.70）
      2. 对每个窗口，根据场景分布计算先验分数（prior score）：
         prior_score = Σ(场景面积占比 × 该场景下目标出现概率)
      3. 综合评分：
         score = α × prior_score + (1-α) × (显著性/2 + 估算置信度/2)

    【参数】
      α（prior_strength，先验强度）= 0.4：场景先验权重（weight）。
      越大越依赖常识。

    【实验结果】
      发现率 60.3%。消融实验中移除后下降 10.0pp — 与 IOR 并列为组合策略中最关键的组件。
    """

    def __init__(self, max_steps: int = 50, prior_config: ScenePriorConfig = None):
        super().__init__("I2_ScenePriorGuided", max_steps)
        self.prior_cfg = prior_config or ScenePriorConfig()

    def _prior_score(self, scene_dist: Dict[str, float], target: str) -> float:
        """
        计算一个窗口的场景先验分数（prior score）。

        【举例】
          窗口场景分布：{"桌子": 0.6, "地板": 0.4}
          目标：水杯
          先验表：水杯→桌子 0.70, 水杯→厨房台面 0.20, 水杯→地板 0.05, ...
          prior_score = 0.6×0.70 + 0.4×0.05 = 0.44
        """
        priors = self.prior_cfg.object_scene_priors.get(target, {})
        if not priors:
            return 0.5  # 无先验知识 → 均匀假设（uniform prior）

        # 累加：每种场景的面积占比 × 该场景下目标出现概率
        return sum(proportion * priors.get(label, 0.05)
                   for label, proportion in scene_dist.items())

    def search(self, scene: PanoramaScene, grid: WindowGrid) -> TrialResult:
        windows = grid.generate(1.0)
        self._precompute_with_scene(windows, scene)
        result = TrialResult(strategy=self.name, target_category=scene.target_category)
        observed_idx: Set[int] = set()
        target = scene.target_category
        alpha = self.prior_cfg.prior_strength  # 0.4
        stop_thresh = self._stop_thresh()

        # ⭐ 预计算所有窗口的场景先验分数
        prior_scores = {
            w.idx: self._prior_score(self._scene(w.idx), target)
            for w in windows
        }

        for step in range(min(self.max_steps, len(windows))):
            best_w, best_s = None, -1.0

            for w in windows:
                if w.idx in observed_idx:
                    continue

                est_conf = self._estimate_conf(w.idx)

                # ⭐ I2 的核心评分公式
                # α×先验 + (1-α)×(显著性/2 + 置信度估算/2)
                # 当没有观察信息时（est_conf=0），显著性提供引导
                # 当有观察信息后，两者的平均值平衡"看到的东西"和"常识"
                score = (alpha * prior_scores[w.idx] +
                         (1 - alpha) * 0.5 * (self._sal(w.idx) + est_conf))

                if score > best_s:
                    best_s, best_w = score, w

            if best_w is None:
                break

            window = best_w
            if window.idx in observed_idx:
                result.repeat_count += 1
            observed_idx.add(window.idx)

            obs = self._observe(window, scene, step)
            result.observations.append(obs)
            result.confidences.append(obs.detection_confidence)

            if obs.detection_confidence >= stop_thresh:
                result.found_target = obs.found_target
                result.false_stop = not scene.has_target_at(obs.window)
                result.total_steps = step + 1
                return result

        result.total_steps = min(self.max_steps, len(windows))
        return result


# ============================================================================
# I3: 粗到细搜索
# ============================================================================

class CoarseToFineSearcher(BaseSearcher):
    """
    I3 — 粗到细搜索（Coarse-to-Fine Search）

    【为什么需要粗到细？】
      如果每次都是 200×200 的高分辨率窗口，搜索 1600×600 的全景需要很多步。
      但如果先用 100×100 的低分辨率窗口快速"扫一遍"，锁定几个候选区域，
      再用高分辨率仔细确认——就可以节省步数。

      类似于：进图书馆找一本书 →
        先扫视书架区域（粗 coarse）→ 锁定第 3 排书架 → 走近仔细看书脊（细 fine）。

    【两阶段（two-phase）】
      Phase 1（阶段一，40% 预算 budget）: 用 0.5x 粗窗口快速扫描
        - 按显著性排序选择粗窗口（不按顺序，优先看可能有东西的地方）
        - 空间分散（spatial diversity）：避免所有粗窗口集中在同一区域

      Phase 2（阶段二，60% 预算）: 在 Phase 1 高置信度区域精细确认
        - 取 Phase 1 中置信度最高的 top-k（前 k 个）粗窗口
        - 在这些粗窗口附近，用 1.0x/1.5x/2.0x 尺度精细搜索

    【实验结果】
      发现率 52.0%，排名第五。
    """

    def __init__(self, max_steps: int = 50, top_k: int = 5,
                 window_config: WindowConfig = None):
        super().__init__("I3_CoarseToFine", max_steps)
        self.top_k = top_k                     # Phase 2 中使用 top-k 个最佳粗窗口
        self.win_cfg = window_config or WindowConfig()

    def search(self, scene: PanoramaScene, grid: WindowGrid) -> TrialResult:
        result = TrialResult(strategy=self.name, target_category=scene.target_category)
        observed_idx: Set[int] = set()
        stop_thresh = self._stop_thresh()

        # ═══════════════════════════════════════════════════════════════
        # Phase 1（阶段一）: 粗搜索（Coarse Search，40% 预算 budget）
        # ═══════════════════════════════════════════════════════════════

        # 生成 0.5x 尺度的粗窗口（100×100 而非 200×200）
        cw_list = list(grid.generate(self.win_cfg.coarse_scale))

        # 为粗窗口预计算显著性
        for w in cw_list:
            crop = w.crop_from(scene.image)
            sal_score, _ = self.saliency_det.compute(crop)
            self._sal_cache[w.idx] = sal_score
            self._window_centers[w.idx] = w.center

        coarse_budget = max(1, int(self.max_steps * 0.4))  # 20 步
        coarse_results = []  # [(window, confidence), ...]

        # ⭐ 选择粗窗口的策略：显著性排序 + 空间分散（spatial diversity）
        #    不是按网格顺序扫描，而是优先看显著的、且彼此间隔远的区域
        cw_sorted = sorted(cw_list,
                           key=lambda w: self._sal_cache.get(w.idx, 0),
                           reverse=True)
        selected_coarse = []
        used_centers = []

        for w in cw_sorted:
            if len(selected_coarse) >= coarse_budget:
                break

            # 空间分散检查：新窗口距已有已选窗口至少 150px
            too_close = False
            for uc in used_centers:
                if np.sqrt((w.center[0] - uc[0])**2 + (w.center[1] - uc[1])**2) < 150:
                    too_close = True
                    break

            if too_close and len(selected_coarse) < coarse_budget:
                continue  # 太近了，跳过

            selected_coarse.append(w)
            used_centers.append(w.center)

        # 如果空间分散后还不够预算 → 从剩余中补充
        if len(selected_coarse) < coarse_budget:
            for w in cw_sorted:
                if w not in selected_coarse and len(selected_coarse) < coarse_budget:
                    selected_coarse.append(w)

        # 执行粗扫描
        for step, window in enumerate(selected_coarse[:coarse_budget]):
            observed_idx.add(window.idx)

            crop = window.crop_from(scene.image)
            conf, found = self.detector.detect(window, scene)
            self._observed_conf[window.idx] = conf

            obs = ObservationResult(
                window=window, image=crop,
                detection_confidence=conf,
                saliency_score=self._sal_cache.get(window.idx, 0.0),
                found_target=found, step=step,
            )
            result.observations.append(obs)
            result.confidences.append(conf)
            coarse_results.append((window, conf))

            # 粗扫描也有可能直接找到！
            if conf >= stop_thresh:
                result.found_target = found
                result.false_stop = not scene.has_target_at(obs.window)
                result.total_steps = step + 1
                return result

        # ═══════════════════════════════════════════════════════════════
        # Phase 2（阶段二）: 精细搜索（Fine Search，60% 预算）
        # ═══════════════════════════════════════════════════════════════

        fine_budget = self.max_steps - len(coarse_results)

        # 取 Phase 1 中置信度最高的 top-k（前 k 个）粗窗口
        coarse_results.sort(key=lambda x: x[1], reverse=True)
        top_regions = coarse_results[:self.top_k]

        if not top_regions:
            result.total_steps = len(coarse_results)
            return result

        # 在 top-k 粗窗口周围收集精细窗口
        fine_candidates = []
        for scale in self.win_cfg.fine_scales:   # [1.0, 1.5, 2.0]
            for fw in grid.generate(scale):
                if fw.idx in observed_idx:
                    continue        #已经观察过的不看
                # 检查是否与任何 top-k 粗窗口有重叠（IoU > 0.05）
                for cw, _ in top_regions:
                    if fw.iou(cw) > 0.05:       # 哪怕只有 5% IoU（交并比）重叠也算"附近"
                        fine_candidates.append(fw)
                        break

        # 去重（deduplicate）
        seen = set()
        unique_candidates = []
        for w in fine_candidates:
            if w.idx not in seen:
                seen.add(w.idx)
                unique_candidates.append(w)
                # 补充预计算信息
                if w.idx not in self._window_centers:
                    self._window_centers[w.idx] = w.center
                    crop = w.crop_from(scene.image)
                    sal_score, _ = self.saliency_det.compute(crop)
                    self._sal_cache[w.idx] = sal_score

        # 按"估算置信度 + 显著性"排序精细候选
        for w in unique_candidates:
            if w.idx not in self._observed_conf:
                self._observed_conf[w.idx] = 0.0

        unique_candidates.sort(
            key=lambda w: self._estimate_conf(w.idx) +
                          0.2 * self._sal_cache.get(w.idx, 0),
            reverse=True
        )

        # 执行精细搜索
        fine_step = len(coarse_results)
        for w in unique_candidates[:fine_budget]:
            if fine_step >= self.max_steps:
                break
            if w.idx in observed_idx:
                result.repeat_count += 1
            observed_idx.add(w.idx)

            crop = w.crop_from(scene.image)
            conf, found = self.detector.detect(w, scene)
            self._observed_conf[w.idx] = conf

            obs = ObservationResult(
                window=w, image=crop,
                detection_confidence=conf,
                saliency_score=self._sal_cache.get(w.idx, 0.0),
                found_target=found, step=fine_step,
            )
            result.observations.append(obs)
            result.confidences.append(conf)
            fine_step += 1

            if conf >= stop_thresh:
                result.found_target = found
                result.false_stop = not scene.has_target_at(obs.window)
                result.total_steps = fine_step
                return result

        result.total_steps = fine_step
        return result


# ============================================================================
# I4: 短期视觉记忆搜索
# ============================================================================

class VisualMemorySearcher(BaseSearcher):
    """
    I4 — 短期视觉记忆搜索（Visual Memory Search）

    【为什么需要视觉记忆（Visual Memory）？】
      B4 只看"当前"观察到的信息。但如果 5 步前你瞥见了一个可疑区域，
      当时置信度不够高所以没停，5 步后你可能就忘了那个位置。
      视觉记忆让系统不会遗忘"曾经发现过但未确认的高显著性区域"。

    【实现】
      1. 维护像素级"记忆热力图" memory_heatmap
      2. 每次观察后，将窗口的显著性图 × 置信度叠加到热力图
      3. 每步热力图 × 0.85 衰减（decay，模拟记忆消退）
      4. 选择下一个窗口时：
         score = est_conf + 0.4 × memory_score + 0.1 × saliency

    【与 IOR 的对比】
      IOR 说的是"别看重复的地方"（抑制 inhibition），
      Memory 说的是"记住有希望的地方"（增强 enhancement）。
      两者互补（complementary）：IOR 推开，Memory 拉回。

    【实验结果】
      发现率 53.3%，相比 B4 (50.0%) 提升 3.3pp（百分点）。
    """

    def __init__(self, max_steps: int = 50, mem_config: MemoryConfig = None):
        super().__init__("I4_VisualMemory", max_steps)
        self.mem_cfg = mem_config or MemoryConfig()

    def _update_memory(self, heatmap: np.ndarray, window: ObservationWindow,
                       scene: PanoramaScene, confidence: float):
        """
        更新记忆热力图（memory heatmap）：把当前窗口的显著性图叠加到热力图上。

        使用 np.maximum（取最大值）而非加法，防止反复观察同一区域
        导致热力值无限累积（accumulation）。
        """
        # 计算当前窗口的显著性图
        crop = window.crop_from(scene.image)
        _, sal_map = self.saliency_det.compute(crop)
        if sal_map is None:
            return

        # 处理边界（bounds），确保尺寸匹配
        x1, y1, x2, y2 = window.bounds
        h_p = min(scene.height, y2) - max(0, y1)
        w_p = min(scene.width, x2) - max(0, x1)
        if h_p <= 0 or w_p <= 0:
            return

        # 把显著性图 resize（缩放）到窗口实际覆盖的像素尺寸
        sal_r = cv2.resize(sal_map, (w_p, h_p))

        y1c, y2c = max(0, y1), min(scene.height, y2)
        x1c, x2c = max(0, x1), min(scene.width, x2)

        # ⭐ 取最大值合并（element-wise maximum）：hot spot（热点）不会被后续低显著性观察"冲淡"
        heatmap[y1c:y2c, x1c:x2c] = np.maximum(
            heatmap[y1c:y2c, x1c:x2c],
            sal_r * confidence  # 显著性 × 检测置信度 = "记忆强度（memory strength）"
        )

    def search(self, scene: PanoramaScene, grid: WindowGrid) -> TrialResult:
        windows = grid.generate(1.0)
        self._precompute_with_scene(windows, scene)
        result = TrialResult(strategy=self.name, target_category=scene.target_category)
        observed_idx: Set[int] = set()

        # ⭐ 记忆热力图（memory heatmap）：形状 = (600, 1600)，每个像素有"记忆值"
        memory_heatmap = np.zeros((scene.height, scene.width), dtype=np.float32)
        decay = self.mem_cfg.memory_decay    # 0.85
        stop_thresh = self._stop_thresh()

        # ---- 第 0 步 ----
        all_sal = [(i, self._sal(w.idx)) for i, w in enumerate(windows)]
        start_i = max(all_sal, key=lambda x: x[1])[0]
        window = windows[start_i]
        observed_idx.add(window.idx)

        obs = self._observe(window, scene, 0)
        result.observations.append(obs)
        result.confidences.append(obs.detection_confidence)

        # 首次更新记忆
        self._update_memory(memory_heatmap, window, scene, obs.detection_confidence)

        if obs.detection_confidence >= stop_thresh:
            result.found_target = obs.found_target
            result.false_stop = not scene.has_target_at(obs.window)
            result.total_steps = 1
            return result

        # ---- 第 1~N 步 ----
        for step in range(1, min(self.max_steps, len(windows))):
            # ⭐ 每步衰减记忆（decay，旧记忆逐渐消退）
            memory_heatmap *= decay

            best_w, best_s = None, -1.0

            for w in windows:
                if w.idx in observed_idx:
                    continue

                # 边界检查（bounds check）
                x1, y1, x2, y2 = w.bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(scene.width, x2), min(scene.height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # 该窗口在记忆热力图中的平均值
                mem_score = np.mean(memory_heatmap[y1:y2, x1:x2])

                est_conf = self._estimate_conf(w.idx)

                # ⭐ I4 的核心评分公式
                # est_conf    → 当前已观察信息的空间插值
                # mem_score   → 记忆中的"未解决高显著性区域"
                # saliency    → 视觉突出度（辅助）
                score = est_conf + 0.4 * mem_score + 0.1 * self._sal(w.idx)

                if score > best_s:
                    best_s, best_w = score, w

            if best_w is None:
                break

            window = best_w
            if window.idx in observed_idx:
                result.repeat_count += 1
            observed_idx.add(window.idx)

            obs = self._observe(window, scene, step)
            result.observations.append(obs)
            result.confidences.append(obs.detection_confidence)

            # 更新记忆
            self._update_memory(memory_heatmap, window, scene, obs.detection_confidence)

            if obs.detection_confidence >= stop_thresh:
                result.found_target = obs.found_target
                result.false_stop = not scene.has_target_at(obs.window)
                result.total_steps = step + 1
                return result

        result.total_steps = min(self.max_steps, len(windows))
        return result


# ============================================================================
# C1: 组合策略 — 融合所有创新
# ============================================================================

class CombinedSearcher(BaseSearcher):
    """
    Combined — 组合策略（Combined Searcher）

    【融合了什么？】
      - I2 的场景先验（Scene Prior，用常识指导搜索方向）
      - I1 的返回抑制（IOR，避免在已观察区域反复看）
      - I4 的视觉记忆（Visual Memory，不忘有希望的未确认区域）

    【评分公式（核心）】
      score = α × prior                          # 场景先验（40%）
            + (1-α) × (
                conf_ior                          # 置信度 × (1 - IOR抑制)
                + 0.3 × memory_score             # 记忆热力
                + 0.1 × saliency                 # 显著性辅助
              )

    【实验结果】
      发现率 69.3%，平均仅 11.4 步 — 效率最优。
      消融实验（ablation study）证明"场景先验"和 IOR 是最关键的组件（移除后各 -10.0pp）。
    """

    def __init__(self, max_steps: int = 50, prior_config: ScenePriorConfig = None,
                 mem_config: MemoryConfig = None):
        super().__init__("Combined", max_steps)
        self.prior_cfg = prior_config or ScenePriorConfig()
        self.mem_cfg = mem_config or MemoryConfig()
        self.ior_strength = 0.6

    def _prior_score(self, scene_dist: Dict[str, float], target: str) -> float:
        """同 I2：计算场景先验分数（prior score）"""
        priors = self.prior_cfg.object_scene_priors.get(target, {})
        if not priors:
            return 0.5
        return sum(proportion * priors.get(label, 0.05)
                   for label, proportion in scene_dist.items())

    def _update_memory(self, heatmap: np.ndarray, window: ObservationWindow,
                       scene: PanoramaScene, confidence: float):
        """同 I4：更新记忆热力图（memory heatmap）"""
        crop = window.crop_from(scene.image)
        _, sal_map = self.saliency_det.compute(crop)
        if sal_map is None:
            return
        x1, y1, x2, y2 = window.bounds
        h_p = min(scene.height, y2) - max(0, y1)
        w_p = min(scene.width, x2) - max(0, x1)
        if h_p <= 0 or w_p <= 0:
            return
        sal_r = cv2.resize(sal_map, (w_p, h_p))
        y1c, y2c = max(0, y1), min(scene.height, y2)
        x1c, x2c = max(0, x1), min(scene.width, x2)
        heatmap[y1c:y2c, x1c:x2c] = np.maximum(
            heatmap[y1c:y2c, x1c:x2c],
            sal_r * confidence
        )

    def search(self, scene: PanoramaScene, grid: WindowGrid) -> TrialResult:
        windows = grid.generate(1.0)
        self._precompute_with_scene(windows, scene)

        result = TrialResult(strategy=self.name, target_category=scene.target_category)
        observed_idx: Set[int] = set()

        # ---- 三大"状态"（state）----
        observed_count = np.zeros((scene.height, scene.width), dtype=np.float32)  # IOR 观察计数
        memory_heatmap = np.zeros((scene.height, scene.width), dtype=np.float32)  # Memory 热力图

        target = scene.target_category
        alpha = self.prior_cfg.prior_strength  # 0.4
        decay = self.mem_cfg.memory_decay      # 0.85
        stop_thresh = self._stop_thresh()

        # 预计算场景先验
        prior_scores = {
            w.idx: self._prior_score(self._scene(w.idx), target)
            for w in windows
        }

        # ---- 第 0 步：显著性+先验综合最高的窗口 ----
        all_scores = [(i, 0.5 * self._sal(w.idx) + 0.5 * prior_scores[w.idx])
                      for i, w in enumerate(windows)]
        start_i = max(all_scores, key=lambda x: x[1])[0]

        window = windows[start_i]
        observed_idx.add(window.idx)

        # 更新 IOR 观察计数
        x1, y1, x2, y2 = window.bounds
        observed_count[max(0, y1):min(scene.height, y2),
                       max(0, x1):min(scene.width, x2)] += 1.0

        obs = self._observe(window, scene, 0)
        result.observations.append(obs)
        result.confidences.append(obs.detection_confidence)

        # 更新记忆
        self._update_memory(memory_heatmap, window, scene, obs.detection_confidence)

        if obs.detection_confidence >= stop_thresh:
            result.found_target = obs.found_target
            result.false_stop = not scene.has_target_at(obs.window)
            result.total_steps = 1
            return result

        # ---- 第 1~N 步 ----
        for step in range(1, min(self.max_steps, len(windows))):
            # 衰减记忆（decay）
            memory_heatmap *= decay

            best_w, best_s = None, -1.0

            for w in windows:
                if w.idx in observed_idx:
                    continue

                # 边界检查（bounds check）
                bx1, by1, bx2, by2 = w.bounds
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(scene.width, bx2), min(scene.height, by2)
                if bx2 <= bx1 or by2 <= by1:
                    continue

                # ---- 1. IOR（返回抑制）: 已观察比例 ----
                patch = observed_count[by1:by2, bx1:bx2]
                observed_ratio = np.mean(patch > 0)

                # ---- 2. 置信度估算 + IOR 抑制 ----
                est_conf = self._estimate_conf(w.idx)
                conf_ior = est_conf * (1.0 - observed_ratio * self.ior_strength)

                # ---- 3. 场景先验（Scene Prior）----
                prior_s = prior_scores[w.idx]

                # ---- 4. 视觉记忆（Visual Memory）----
                mem_score = np.mean(memory_heatmap[by1:by2, bx1:bx2])

                # ⭐ 组合评分公式（本项目的最终策略 final strategy）:
                score = (alpha * prior_s +                          # 40% 场景先验
                         (1 - alpha) * (                             # 60% 感知+记忆
                             conf_ior +                             # 置信度 × (1-IOR)
                             0.3 * mem_score +                      # 记忆热力
                             0.1 * self._sal(w.idx)                 # 显著性辅助
                         ))

                if score > best_s:
                    best_s, best_w = score, w

            if best_w is None:
                break

            window = best_w
            if window.idx in observed_idx:
                result.repeat_count += 1
            observed_idx.add(window.idx)

            # 更新 IOR 观察计数
            bx1, by1, bx2, by2 = window.bounds
            observed_count[max(0, by1):min(scene.height, by2),
                           max(0, bx1):min(scene.width, bx2)] += 1.0

            obs = self._observe(window, scene, step)
            result.observations.append(obs)
            result.confidences.append(obs.detection_confidence)

            # 更新视觉记忆
            self._update_memory(memory_heatmap, window, scene, obs.detection_confidence)

            if obs.detection_confidence >= stop_thresh:
                result.found_target = obs.found_target
                result.false_stop = not scene.has_target_at(obs.window)
                result.total_steps = step + 1
                return result

        result.total_steps = min(self.max_steps, len(windows))
        return result


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                   七、实验框架                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝


# ============================================================================
# ExperimentRunner — 批量实验运行器
# ============================================================================

class ExperimentRunner:
    """
    管理全部实验流程：生成场景 → 运行所有策略 → 收集指标 → 打印结果。

    【三层嵌套循环（nested loop）】
      for 每个场景:
        for 每个目标类别:
          for 每次 trial（试验）:
            生成场景
            for 每个搜索策略:
              运行搜索 → 记录结果
    """

    def __init__(self, exp_config: ExperimentConfig = None):
        self.cfg = exp_config or config
        # 策略名 → 指标追踪器（metrics tracker）
        self.all_trackers: Dict[str, MetricsTracker] = defaultdict(MetricsTracker)

    def _create_searchers(self) -> List[BaseSearcher]:
        """创建所有 9 个搜索策略的实例（instance）"""
        ms = self.cfg.max_steps
        return [
            # Baseline（基线策略）
            SequentialScanSearcher(max_steps=ms),
            RandomScanSearcher(max_steps=ms),
            SaliencyGuidedSearcher(max_steps=ms),
            DetectionConfidenceSearcher(max_steps=ms),
            # Innovation（创新策略）
            InhibitionOfReturnSearcher(max_steps=ms),
            ScenePriorGuidedSearcher(max_steps=ms),
            CoarseToFineSearcher(max_steps=ms),
            VisualMemorySearcher(max_steps=ms),
            # Combined（组合策略）
            CombinedSearcher(max_steps=ms),
        ]

    def run(self) -> Dict[str, Dict]:
        """
        运行所有实验。

        返回: {策略名: {指标名: 值, ...}, ...}
        """
        searchers = self._create_searchers()
        n_strategies = len(searchers)
        n_categories = len(self.cfg.target_categories)
        n_scenes = self.cfg.n_synthetic_scenes
        n_trials = self.cfg.n_trials
        total = n_strategies * n_categories * n_scenes * n_trials

        print(f"{'='*70}")
        print(f"  基于全景图像的无地图主动目标搜索 -- 实验评估（evaluation）")
        print(f"  策略数: {n_strategies} | 目标类别: {n_categories}")
        print(f"  场景数: {n_scenes} | 每场景trial数: {n_trials}")
        print(f"  总实验数: {total}")
        print(f"{'='*70}\n")

        # 初始化每个策略的 tracker（追踪器）
        for s in searchers:
            self.all_trackers[s.name] = MetricsTracker()

        trial_id = 0

        # 三层嵌套循环（nested loop）
        for scene_idx in range(n_scenes):
            scene_seed = self.cfg.random_seed + scene_idx * 1000

            for category in self.cfg.target_categories:
                for trial in range(n_trials):
                    # 设置随机种子（seed）确保可复现（reproducible）
                    trial_seed = scene_seed + trial * 100 + _category_seed(category)
                    np.random.seed(trial_seed)

                    # 生成场景
                    gen = SyntheticSceneGenerator(seed=trial_seed)
                    scene = gen.generate(category)
                    grid = WindowGrid(scene.width, scene.height, WindowConfig())

                    # 所有策略在同一个场景上搜索
                    for searcher in searchers:
                        result = searcher.search(scene, grid)
                        result.trial_id = trial_id
                        self.all_trackers[searcher.name].add(result)
                        trial_id += 1

                # 进度报告（progress report）
                progress = trial_id / total * 100
                print(f"  进度: {trial_id}/{total} ({progress:.0f}%)")

        print(f"  全部 {total} 次试验完成!\n")
        return self._compute_summaries()

    def _compute_summaries(self) -> Dict[str, Dict]:
        """汇总所有策略的指标（summarize）"""
        return {name: tracker.summary()
                for name, tracker in self.all_trackers.items()}

    def print_results(self, summaries: Dict[str, Dict]):
        """
        打印格式化的实验结果表格。

        输出四个部分：
          1. Baseline（基线）策略表格
          2. 创新策略表格
          3. Combined vs 最佳 Baseline 对比
          4. 综合排名（ranking）
        """
        baseline = [
            "B1_SequentialScan", "B2_RandomScan",
            "B3_SaliencyGuided", "B4_DetectionConfidence"
        ]
        innovation = [
            "I1_InhibitionOfReturn", "I2_ScenePriorGuided",
            "I3_CoarseToFine", "I4_VisualMemory"
        ]
        combined = "Combined"

        # 格式化辅助函数
        def fp(v):  # 百分比格式化（format percentage）
            return "   N/A" if np.isnan(v) else f"{v*100:6.1f}%"

        def fv(v):  # 数值格式化（format value）
            return "   N/A" if np.isnan(v) else f"{v:6.1f}"

        header = (f"{'Strategy':<28} {'Found%':>8} {'AvgSteps':>8} "
                  f"{'MedSteps':>8} {'Repeat%':>8} {'False%':>8}")
        sep = "-" * 74

        # 打印 Baseline 和创新策略的表格
        for title, names in [("Baseline Strategies", baseline),
                              ("Innovation Strategies", innovation)]:
            print(f"\n{'='*74}")
            print(f"  [{title}]")
            print(sep)
            print(header)
            print(sep)
            for name in names:
                s = summaries.get(name, {})
                print(f"  {name:<26} {fp(s.get('discovery_rate', 0))} "
                      f"{fv(s.get('avg_steps', 0))} "
                      f"{fv(s.get('median_steps', 0))} "
                      f"{fp(s.get('repeat_rate', 0))} "
                      f"{fp(s.get('false_stop_rate', 0))}")
            print(sep)

        # Combined vs 最佳 Baseline
        print(f"\n{'='*74}")
        print(f"  [Combined vs Best Baseline]")
        print(sep)
        print(header)
        print(sep)
        best_base = max(baseline,
                        key=lambda n: summaries.get(n, {}).get("discovery_rate", 0))
        for name in [best_base, combined]:
            s = summaries.get(name, {})
            tag = " <-- best baseline" if name == best_base else " <-- combined"
            print(f"  {name:<26} {fp(s.get('discovery_rate', 0))} "
                  f"{fv(s.get('avg_steps', 0))} "
                  f"{fv(s.get('median_steps', 0))} "
                  f"{fp(s.get('repeat_rate', 0))} "
                  f"{fp(s.get('false_stop_rate', 0))}{tag}")
        print(sep)

        # 综合排名
        print(f"\n{'='*74}")
        print(f"  [Ranking by Discovery Rate]")
        print(sep)
        ranked = sorted(summaries.items(),
                        key=lambda x: x[1].get("discovery_rate", 0), reverse=True)
        for rank, (name, s) in enumerate(ranked, 1):
            dr = fp(s.get("discovery_rate", 0))
            avg = fv(s.get("avg_steps", 0))
            medal = {1: "[1]", 2: "[2]", 3: "[3]"}.get(rank, f"  {rank}.")
            print(f"  {medal} {name:<26} Found: {dr}  AvgSteps: {avg}")
        print("=" * 74)


# ============================================================================
# 消融实验（Ablation Study）— 逐步移除组件，量化贡献
# ============================================================================

def run_ablation_study(exp_cfg: ExperimentConfig = None):
    """
    消融实验（Ablation Study）：从 Combined 策略中每次移除一个创新组件，
    观察性能变化。

    【实验设计】
      - Combined (all on)     : 所有组件启用（基准 baseline）
      - - Scene Prior         : 关闭场景先验（prior_strength=0）
      - - IOR                 : 关闭返回抑制（ior_strength=0）
      - - Visual Memory       : 关闭视觉记忆（decay=0, capacity=0）

    【解读】
      如果移除组件 X 后性能显著下降 → X 是关键组件（critical component）。
      如果移除组件 X 后性能不变 → X 在该配置下贡献有限。
    """
    print(f"\n{'='*74}")
    print(f"  [Ablation Study -- Remove One Component at a Time]")
    print("=" * 74)

    cfg = exp_cfg or config
    n_scenes = max(1, cfg.n_synthetic_scenes // 2)  # 减半加速
    n_trials = max(1, cfg.n_trials // 2)

    # 定义四个变体（variant）
    variants = [
        ("Combined (all on)", CombinedSearcher(max_steps=cfg.max_steps)),

        ("- Scene Prior",
         CombinedSearcher(max_steps=cfg.max_steps,
                          prior_config=ScenePriorConfig(prior_strength=0.0))),
                        # prior_strength=0：把"场景先验"的权重设为 0，等于关闭场景先验
        ("- IOR", CombinedSearcher(max_steps=cfg.max_steps)),  # 关闭 IOR 的处理在下方循环里（把 ior_strength 设为 0）

        ("- Visual Memory",
         CombinedSearcher(max_steps=cfg.max_steps,
                          mem_config=MemoryConfig(memory_decay=0.0, memory_capacity=0))),
    ]

    results = {}
    for vname, searcher in variants:
        # 特殊处理：关闭 IOR
        if "IOR" in vname and "all on" not in vname:
            searcher.ior_strength = 0.0

        tracker = MetricsTracker()

        for scene_idx in range(n_scenes):
            scene_seed = cfg.random_seed + scene_idx * 1000 + 99999
            for category in cfg.target_categories[:3]:  # 只用前 3 个类别加速
                for trial in range(n_trials):
                    trial_seed = scene_seed + trial * 100 + _category_seed(category)
                    np.random.seed(trial_seed)
                    gen = SyntheticSceneGenerator(seed=trial_seed)
                    scene = gen.generate(category)
                    grid = WindowGrid(scene.width, scene.height, WindowConfig())
                    result = searcher.search(scene, grid)
                    tracker.add(result)
    
        results[vname] = tracker.summary()

    # 打印消融结果
    header = (f"{'Variant':<28} {'Found%':>8} {'AvgSteps':>8} "
              f"{'Repeat%':>8} {'False%':>8}")
    sep = "-" * 70
    print(sep)
    print(header)
    print(sep)

    base_dr = results.get("Combined (all on)", {}).get("discovery_rate", 0)
    for name, s in results.items():
        dr = s.get("discovery_rate", 0)
        delta = dr - base_dr if "all on" not in name else 0
        d_str = f" ({delta*100:+.1f}%)" if delta != 0 else ""
        print(f"  {name:<26} {dr*100:6.1f}%{d_str} "
              f"{s.get('avg_steps', float('nan')):8.1f} "
              f"{s.get('repeat_rate', 0)*100:6.1f}% "
              f"{s.get('false_stop_rate', 0)*100:6.1f}%")
    print(sep)
    print("  (Negative delta = performance drops without this component)")
    print("=" * 74)


# ============================================================================
# 主入口（entry point）
# ============================================================================

def main():
    """
    程序入口。

    【运行流程】
      1. 创建实验配置（快速模式：少量 trial 和场景）
      2. 运行所有 9 个搜索策略
      3. 打印对比结果表格
      4. 运行消融实验（ablation study）
      5. 报告总耗时
    """
    t_start = time.time()

    # ---- 完整实验配置（规模从 config.py 读取，改 config.py 即可调整） ----
    exp_config = ExperimentConfig(
        max_steps=config.max_steps,                 # 50 步
        n_trials=config.n_trials,                   # 每场景 10 次重复（取自 config.py）
        n_synthetic_scenes=config.n_synthetic_scenes,  # 5 个不同场景（取自 config.py）
        target_categories=config.target_categories, # 6 个目标类别
        random_seed=config.random_seed,             # 42
    )

    np.random.seed(exp_config.random_seed)

    # ---- 运行实验 ----
    runner = ExperimentRunner(exp_config)
    summaries = runner.run()
    runner.print_results(summaries)

    # ---- 消融实验（Ablation Study） ----
    run_ablation_study(exp_config)

    t_elapsed = time.time() - t_start
    print(f"\nTotal time: {t_elapsed:.1f}s")
    print("Tip: Increase n_synthetic_scenes and n_trials for full experiment")


# ============================================================================
# Python 入口
# ============================================================================

if __name__ == "__main__":
    main()
