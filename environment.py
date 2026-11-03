"""
=============================================================================
环境模块 — 全景场景、观察窗口与合成场景生成
=============================================================================

【本文件的作用】
  定义"搜索环境"的所有组件：
    1. 全景图是什么样的？        → PanoramaScene
    2. 每次能看多大一块？        → ObservationWindow
    3. 可以选哪些窗口来看？      → WindowGrid（候选窗口网格）
    4. 看完后得到什么信息？      → ObservationResult
    5. 测试场景怎么造出来的？    → SyntheticSceneGenerator

【类比理解】
  把全景图想象成一张大地图，窗口相当于手电筒的光斑——
  你只能看到光斑照亮的那一小块区域，需要决定手电筒往哪照。

【依赖关系】
  本模块只依赖 config.py，不依赖 main.py。
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import cv2
from config import (
    PanoramaConfig, WindowConfig, ScenePriorConfig
)


# ============================================================================
# ObservationWindow — 观察窗口（"手电筒的光斑"）
# ============================================================================

@dataclass
class ObservationWindow:
    """
    表示全景图中的一个局部观察窗口。

    【什么是窗口？】
      全景图是 1600×600 的大图。但我们模拟的是"视场受限相机"——
      每次只能看到 200×200 的一小块。这个类就描述了"哪一块"。

    【属性说明】
      - x, y：窗口左上角在全景图中的坐标
      - width, height：窗口的大小（通常 200×200）
      - scale（缩放比例）：1.0 = 原始大小，0.5 = 缩小一半（粗搜索用）
      - idx（索引）：在候选窗口列表中的编号（唯一标识符 identifier）

    【示例】
      ObservationWindow(x=0, y=0, width=200, height=200, idx=0)
      → 全景图左上角的 200×200 区域

      ObservationWindow(x=100, y=100, width=200, height=200, idx=12)
      → 向右下各偏移 100px 的 200×200 区域（和上一个窗口有重叠 overlap）
    """
    x: int                # 左上角 x 坐标（像素）
    y: int                # 左上角 y 坐标（像素）
    width: int            # 窗口宽度
    height: int           # 窗口高度
    scale: float = 1.0    # 缩放比例。0.5 = 低分辨率快速扫描，1.0 = 原始分辨率
    idx: int = -1         # 候选网格中的唯一编号（index）。相当于窗口的"身份证号"

    # ---- 以下是用 @property（属性装饰器）装饰的计算属性（像属性一样使用，但会动态计算） ----

    @property
    def center(self) -> Tuple[float, float]:
        """窗口中心点坐标。用于计算"窗口到目标的距离"."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> int:
        """窗口面积（像素数）。用于计算 IoU（Intersection over Union，交并比）."""
        return self.width * self.height

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """窗口的四条边界（bounds）：(x1, y1, x2, y2) = (左, 上, 右, 下)"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    # ---- 窗口间关系计算 ----

    def iou(self, other: "ObservationWindow") -> float:
        """
        计算与另一个窗口的 IoU（Intersection over Union，交并比）。

        【什么是 IoU？】
          IoU = 两个窗口重叠面积（intersection）/ 两个窗口总面积（union）
          完全重叠 → 1.0；完全不重叠 → 0.0

        【用途】
          粗到细搜索（Coarse-to-Fine Search）时，判断精细窗口是否"在粗窗口附近"：
          如果 IoU > 0.05 → 认为在附近，值得精细查看。
        """
        # 计算交集矩形的四条边
        x1 = max(self.x, other.x)                          # 交集左边界 = 两者左边界的较大者
        y1 = max(self.y, other.y)                          # 交集上边界 = 两者上边界的较大者
        x2 = min(self.x + self.width, other.x + other.width)   # 交集右边界
        y2 = min(self.y + self.height, other.y + other.height) # 交集下边界

        inter = max(0, x2 - x1) * max(0, y2 - y1)          # 交集面积。max(0,...) 处理无重叠情况
        union = self.area + other.area - inter              # 并集面积
        return inter / union if union > 0 else 0.0

    def overlap_ratio(self, other: "ObservationWindow") -> float:
        """
        计算本窗口被另一个窗口覆盖的比例（overlap ratio）。

        【与 IoU 的区别】
          IoU 是"对称"的：不论 A.iou(B) 还是 B.iou(A)，结果相同。
          overlap_ratio 是"非对称"的：A 被 B 盖住了多少。

        【用途】
          IOR（Inhibition of Return，返回抑制）策略中判断"这个区域被看过多少了"。
        """
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        return inter / self.area if self.area > 0 else 0.0

    def crop_from(self, image: np.ndarray) -> np.ndarray:
        """
        从全景图中裁剪出这个窗口对应的图像区域。

        【用途】
          每次"观察"时调用：把全景图中属于这个窗口的像素取出来，
          交给模拟检测器 / 显著性检测器处理。

        【边界处理（boundary handling）】
          如果窗口部分超出全景图边界 → clamp（裁剪/限制）到边界内 → resize（缩放）回标准大小。
          正常情况下（候选窗口都在边界内）不会触发。
        """
        h, w = image.shape[:2]            # 全景图的实际尺寸
        x1, y1, x2, y2 = self.bounds

        # clamp（裁剪到边界内）：确保裁剪坐标不超出图像边界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = image[y1:y2, x1:x2]        # numpy 切片（slicing）：image[行范围, 列范围]

        # 如果裁剪结果尺寸不对（窗口部分越界）→ resize 回标准尺寸
        if crop.shape[0] != self.height or crop.shape[1] != self.width:
            crop = cv2.resize(crop, (self.width, self.height))
        return crop


# ============================================================================
# ObservationResult — 一次观察的结果（"看完之后知道了什么"）
# ============================================================================

@dataclass
class ObservationResult:
    """
    一次观察后获得的所有信息。

    【数据流】
      Searcher._observe() → ObservationResult → 存入 TrialResult.observations

    【各字段含义】
      - window：看了哪个窗口（位置信息）
      - image：看到的图像内容（numpy 数组 / ndarray）
      - detection_confidence：检测器给出的"这里有目标"概率（0~1）
      - saliency_score：显著性检测给出的"这里有多显眼"（0~1）
      - scene_labels：场景标签（label）分布（如 {"桌子": 0.6, "地板": 0.4}）
      - found_target：是否满足"找到目标"的条件
      - step：这是第几步观察
    """
    window: ObservationWindow                 # 观察了哪个窗口
    image: np.ndarray                         # 看到的图像（RGB，numpy 数组）
    detection_confidence: float               # 目标检测置信度（confidence）[0, 1]
    saliency_score: float                     # 该窗口的显著性分数（saliency score）[0, 1]
    saliency_map: Optional[np.ndarray] = None # 显著性热力图（saliency map，像素级，可选）
    scene_labels: Optional[Dict[str, float]] = None  # 场景标签 → 面积占比
    found_target: bool = False                # 是否判定"找到目标"
    step: int = 0                             # 第几步观察（从 0 开始）

    def __hash__(self):
        """让 ObservationResult 可以作为字典（dict）的 key 或放入 set（集合）"""
        return hash((self.window.idx, self.step))


# ============================================================================
# SyntheticSceneGenerator — 合成场景生成器
# ============================================================================

class SyntheticSceneGenerator:
    """
    生成带有场景标注的合成全景图，用于测试搜索策略。

    【为什么用合成图？】
      1. 不需要真实数据集（dataset，省去收集标注的麻烦）
      2. 目标位置已知（方便评估搜索策略是否真的找到了）
      3. 场景标签（label）完美准确（方便测试场景先验策略）
      4. 可控制难度（改变目标大小、噪声等）

    【生成的场景长什么样？】
      类似简化的 Minecraft 俯视图：
      - 上半部：天空（蓝色）+ 建筑/草地/道路（中间过渡带）
      - 下半部：道路/草地/停车场
      - 随机散布"桌子"块（棕色），桌子下面有"地板"（米色）
      - 在合理位置放置目标（如"水杯"放在桌子上）

    【场景颜色参考】
      天空=天蓝  建筑=灰色  道路=深灰  草地=绿色
      桌子=棕色  地板=米色  墙壁=浅灰  沙发=紫色
    """

    # ---- 场景类型 → RGB 颜色 ----
    SCENE_COLORS = {
        "天空":   (135, 206, 235),   # 天蓝色
        "建筑":   (180, 180, 180),   # 灰色
        "道路":   (80, 80, 80),      # 深灰色
        "草地":   (60, 180, 60),     # 绿色
        "桌子":   (139, 90, 43),     # 棕色（木纹）
        "地板":   (200, 180, 150),   # 米色
        "墙壁":   (220, 210, 190),   # 浅灰白
        "沙发":   (100, 80, 120),    # 紫色
        "厨房台面": (180, 160, 140), # 浅棕色
        "床":     (160, 140, 180),   # 淡紫色
        "书架":   (120, 80, 50),     # 深棕色
        "停车场": (100, 100, 100),   # 暗灰色
    }

    # ---- 目标类别 → 显示颜色（在图片上画圆点表示） ----
    TARGET_COLORS = {
        "水杯": (0, 150, 255),    # 橙色（OpenCV 用 BGR（蓝绿红）顺序，这里是 RGB（红绿蓝））
        "汽车": (255, 100, 50),   # 偏红色
        "椅子": (0, 200, 100),    # 绿色
        "电脑": (200, 200, 50),   # 黄色
        "书本": (255, 200, 100),  # 橙色
        "人":   (255, 100, 200),  # 粉色
    }

    # ---- 每种目标只能出现在特定的场景区域中 ----
    # 例如："水杯"不会出现在"天空"里，"汽车"不会出现在"桌子"上
    TARGET_VALID_SCENES = {
        "水杯": ["桌子", "厨房台面"],
        "汽车": ["道路", "停车场"],
        "椅子": ["桌子", "地板"],
        "电脑": ["桌子", "床"],
        "书本": ["桌子", "书架"],
        "人":   ["道路", "草地", "桌子", "沙发"],
    }

    def __init__(self, width: int = 1600, height: int = 600, seed: int = 42):
        """
        初始化生成器。

        参数:
          width: 全景图宽度（默认 1600）
          height: 全景图高度（默认 600）
          seed: 随机种子（seed，固定后可复现相同场景）
        """
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)  # RandomState（随机状态）是独立的随机数生成器

    def _generate_scene_layout(self) -> np.ndarray:
        """
        生成场景标签图（每个像素属于哪种场景类型）。

        【方法】
          1. 把全景图分成 8×3 = 24 个大方块（每块 200×200）
          2. 按行分配场景类型：
             第 0 行（最上方）= 天空
             第 1 行（中间）  = 建筑/草地/道路（随机）
             第 2 行（最下方）= 道路/草地/停车场（随机）
          3. 随机在非天空行放 2-5 个"桌子"块
          4. 桌子下方放"地板"
          5. 放大回原始分辨率

        返回:
          形状为 (600, 1600) 的整数数组（ndarray），每个值代表一种场景类型。
        """
        w_div = 8                           # 水平分成 8 块
        h_div = 3                           # 垂直分成 3 块
        bw, bh = self.width // w_div, self.height // h_div  # 每块 200×200
        labels = np.zeros((h_div, w_div), dtype=int)
        scene_names = list(self.SCENE_COLORS.keys())

        # 逐块分配场景类型
        for i in range(h_div):              # i = 行号（0, 1, 2）
            for j in range(w_div):          # j = 列号（0~7）
                if i == 0:
                    # 最上面一行 = 天空
                    labels[i, j] = scene_names.index("天空")
                elif i == 1:
                    # 中间一行 = 随机选建筑/草地/道路（模拟地平线过渡带）
                    labels[i, j] = self.rng.choice([
                        scene_names.index("建筑"),
                        scene_names.index("草地"),
                        scene_names.index("道路"),
                    ])
                else:
                    # 最下面一行 = 随机选道路/草地/停车场（地面）
                    labels[i, j] = self.rng.choice([
                        scene_names.index("道路"),
                        scene_names.index("草地"),
                        scene_names.index("停车场"),
                    ])

        # 随机添加"桌子"区域（模拟室内元素）
        n_tables = self.rng.randint(2, 5)   # 随机 2~4 张桌子
        for _ in range(n_tables):
            ti = self.rng.randint(1, h_div) # 桌子行号（不能是天空行）
            tj = self.rng.randint(0, w_div) # 桌子列号
            labels[ti, tj] = scene_names.index("桌子")
            # 桌子下面放地板（模拟桌子放在地板上）
            if ti + 1 < h_div:
                labels[ti + 1, tj] = scene_names.index("地板")

        # 把 8×3 的粗标签放大到 1600×600
        full_labels = np.zeros((self.height, self.width), dtype=int)
        for i in range(h_div):
            for j in range(w_div):
                y1, y2 = i * bh, min((i + 1) * bh, self.height)
                x1, x2 = j * bw, min((j + 1) * bw, self.width)
                full_labels[y1:y2, x1:x2] = labels[i, j]

        return full_labels

    def _labels_to_image(self, labels: np.ndarray) -> np.ndarray:
        """
        把场景标签图转换为 RGB 彩色图像。

        【步骤】
          1. 根据 SCENE_COLORS 给每种场景类型涂色
          2. 添加随机噪声（noise，±15）让颜色不那么单调
          3. GaussianBlur（高斯模糊）平滑过渡

        返回: (600, 1600, 3) 的 RGB 图像
        """
        scene_names = list(self.SCENE_COLORS.keys())
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 逐场景类型涂色
        for idx, name in enumerate(scene_names):
            mask = labels == idx              # 该场景类型的像素位置（布尔掩码 boolean mask）
            color = np.array(self.SCENE_COLORS[name], dtype=np.uint8)
            image[mask] = color               # 广播赋值（broadcasting）

        # 添加随机纹理噪声（让色块看起来不那么"塑料"）
        noise = self.rng.randint(-15, 16, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)#重要解释

        # 高斯模糊（Gaussian Blur，让色块边界柔和过渡）
        image = cv2.GaussianBlur(image, (5, 5), 1.5)
        return image

    def _place_targets(self, image: np.ndarray, labels: np.ndarray,
                       target_category: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        在合理位置放置目标物体（画彩色圆点）。

        【逻辑】
          1. 找到目标类别"允许出现"的场景区域
          2. 从这些区域中随机选 1-3 个位置
          3. 画实心圆（filled circle，彩色）+ 黑色边框（border）
          4. 确保目标之间不重叠（距离 > 40px）

        返回:
          - 带目标的图像
          - 目标中心坐标列表 [(cx, cy), ...]
        """
        # 找出目标可以放在哪些场景类型中
        valid_scenes = self.TARGET_VALID_SCENES.get(target_category, ["桌子"])
        scene_names = list(self.SCENE_COLORS.keys())
        valid_indices = [scene_names.index(s) for s in valid_scenes if s in scene_names]

        # 筛选出有效场景对应的所有像素位置
        target_mask = np.isin(labels, valid_indices)
        valid_ys, valid_xs = np.where(target_mask)     # 所有合法位置的 (y, x) 坐标

        positions = []
        n_targets = self.rng.randint(1, 4)             # 每个场景 1~3 个目标
        img = image.copy()

        if len(valid_ys) == 0:                         # 没有合法位置（极端情况）
            return img, positions

        target_color = self.TARGET_COLORS.get(target_category, (255, 0, 0))

        for _ in range(n_targets):
            if len(valid_ys) == 0:
                break
            # 随机选一个合法位置
            idx = self.rng.randint(0, len(valid_ys))
            cy, cx = valid_ys[idx], valid_xs[idx]

            # 画目标：实心圆（filled circle，半径 8-18px）+ 黑色边框
            radius = self.rng.randint(8, 18)
            cv2.circle(img, (cx, cy), radius, target_color, -1)   # -1 = 实心填充（filled）
            cv2.circle(img, (cx, cy), radius, (0, 0, 0), 2)       # 2px 黑色边框（border）
            positions.append((cx, cy))

            # 从候选位置中移除距离当前目标太近的像素（避免目标挤在一起）
            dists = np.sqrt((valid_ys - cy) ** 2 + (valid_xs - cx) ** 2)
            valid_ys = valid_ys[dists > 40]
            valid_xs = valid_xs[dists > 40]

        return img, positions

    def generate(self, target_category: str) -> "PanoramaScene":
        """
        主入口：生成一个包含指定目标类别的完整合成场景。

        【调用链（call chain）】
          generate("水杯")
            → _generate_scene_layout()    # 生成场景标签图
            → _labels_to_image()          # 转为彩色图像
            → _place_targets()            # 放置目标圆点
            → 构建 PanoramaScene 返回

        参数:
          target_category: 目标类别名（如 "水杯"）

        返回:
          一个完整的 PanoramaScene 对象
        """
        labels = self._generate_scene_layout()
        image = self._labels_to_image(labels)
        image_with_targets, target_positions = self._place_targets(image, labels, target_category)

        # 构建场景名 → 二值掩码（binary mask）的映射（用于快速查询某像素属于什么场景）
        scene_names = list(self.SCENE_COLORS.keys())
        scene_label_map = {scene_names[idx]: (labels == idx)
                          for idx in range(len(scene_names))}

        return PanoramaScene(
            image=image_with_targets,
            scene_labels=labels,
            scene_label_map=scene_label_map,
            target_positions=target_positions,
            target_category=target_category,
        )


# ============================================================================
# PanoramaScene — 全景场景（"搜索的地图"）
# ============================================================================

@dataclass
class PanoramaScene:
    """
    全景场景的完整表示。

    【这个类做什么？】
      储存全景图的所有信息，并提供查询接口（API）：
        - 目标在窗口里吗？              → has_target_at()
        - 窗口中心离目标多远？          → distance_to_nearest_target()
        - 这个窗口是什么场景？          → get_scene_distribution()

    【为什么不直接读像素？】
      因为合成场景有"真值标注"（ground truth）——
      我们知道每个像素属于什么场景、目标在什么位置。
      在真实应用中，这些信息来自语义分割模型（Semantic Segmentation，如 DeepLab）
      和目标检测模型（如 YOLO），但在仿真中我们直接从数据结构读取。
    """
    image: np.ndarray                           # RGB 全景图，形状 (600, 1600, 3)
    scene_labels: np.ndarray                    # 逐像素场景标签，形状 (600, 1600)
    scene_label_map: Dict[str, np.ndarray]     # 场景名 → 二值掩码（binary mask）。如 {"桌子": [[True,False,...],...]}
    target_positions: List[Tuple[int, int]]    # 目标中心坐标列表。如 [(800, 300), (400, 450)]
    target_category: str = ""                  # 目标类别名。如 "水杯"

    @property
    def shape(self) -> Tuple[int, int, int]:
        """返回 (高度, 宽度, 通道数 channel)，如 (600, 1600, 3)"""
        return self.image.shape

    @property
    def width(self) -> int:
        """全景图宽度（像素）"""
        return self.image.shape[1]

    @property
    def height(self) -> int:
        """全景图高度（像素）"""
        return self.image.shape[0]

    def has_target_at(self, window: ObservationWindow) -> bool:
        """
        检查窗口中是否包含目标。

        【用途】
          模拟检测器的"保底逻辑"：如果窗口确实覆盖了目标，
          即使用距离算出的置信度偏低，也强制提升置信度。
          确保目标不会因为噪声被完全漏掉。

        【实现】
          遍历所有目标的坐标，检查是否有任何一个落在窗口范围内。
        """
        for cx, cy in self.target_positions:
            if (window.x <= cx < window.x + window.width and
                window.y <= cy < window.y + window.height):
                return True
        return False

    def distance_to_nearest_target(self, window: ObservationWindow) -> float:
        """
        计算窗口中心到最近目标的欧几里得距离（Euclidean distance）。

        【用途】
          这是模拟检测器的核心输入：
          confidence = peak × exp(-distance² / (2 × sigma²))
          距离越近 → 置信度越高。

        【返回值】
          浮点数（像素）。如果没有目标，返回 inf（infinity，无穷大）。
        """
        wcx, wcy = window.center                # 窗口中心坐标
        min_dist = float("inf")
        for cx, cy in self.target_positions:
            # 欧几里得距离：sqrt((x1-x2)² + (y1-y2)²)
            dist = np.sqrt((wcx - cx) ** 2 + (wcy - cy) ** 2)
            min_dist = min(min_dist, dist)
        return min_dist

    def get_scene_distribution(self, window: ObservationWindow) -> Dict[str, float]:
        """
        获取窗口内各场景标签的面积占比（distribution，分布）。

        【返回值示例】
          {"桌子": 0.60, "地板": 0.30, "墙壁": 0.10}
          意思是这个窗口内 60% 是桌子，30% 是地板，10% 是墙壁。

        【用途】
          - 场景先验策略（I2）：用场景分布 × 先验概率来评估窗口的"价值"
          - 理解检测结果："为什么在这个窗口检测置信度低？因为它是天空区域"
        """
        x1, y1, x2, y2 = window.bounds

        # clamp 到图像边界内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)

        crop_labels = self.scene_labels[y1:y2, x1:x2]  # 窗口内的场景标签
        total = crop_labels.size                        # 窗口总像素数
        if total == 0:
            return {}

        # 统计每种场景的面积占比
        dist = {}
        for name, mask in self.scene_label_map.items():
            crop_mask = mask[y1:y2, x1:x2]             # 窗口内的场景掩码（mask）
            dist[name] = float(np.sum(crop_mask)) / total  # True=1, 求和=该场景像素数
        return dist


# ============================================================================
# WindowGrid — 候选窗口网格（"所有可以看的位置"）
# ============================================================================

class WindowGrid:
    """
    在全景图上生成候选观察窗口的网格（grid）。

    【核心功能】
      generate(scale=1.0) → 返回所有候选窗口的列表。

    【举例】
      全景图 1600×600，窗口 200×200，步长（stride）100×100：
        水平方向：(1600-200)/100 + 1 = 15 个窗口
        垂直方向：(600-200)/100 + 1 = 5 个窗口
        总计约 75 个候选窗口（含边界对齐补偿）

    【多尺度（multi-scale）】
      generate(scale=0.5) → 窗口缩小到 100×100，步长 50
        → 生成的候选窗口更多（约 150+ 个），但分辨率低
        用于粗到细搜索（Coarse-to-Fine）的"粗阶段"
    """

    def __init__(self, panorama_width: int, panorama_height: int,
                 window_config: WindowConfig):
        self.pano_w = panorama_width
        self.pano_h = panorama_height
        self.win_w = window_config.width        # 200
        self.win_h = window_config.height       # 200
        self.stride_x = window_config.stride_x  # 100
        self.stride_y = window_config.stride_y  # 100
        self.coarse_scale = window_config.coarse_scale  # 0.5
        self.fine_scales = window_config.fine_scales    # [1.0, 1.5, 2.0]

    def generate(self, scale: float = 1.0) -> List[ObservationWindow]:
        """
        生成指定尺度的候选窗口列表。

        【算法】
          1. 根据 scale（缩放比例）缩放窗口尺寸和步长
          2. 从左上角开始，按步长滑动窗口（sliding window）
          3. 对右边界和底边界做对齐补偿（确保全景图边缘也被覆盖）

        【参数】
          scale = 1.0：原始大小（200×200 窗口）
          scale = 0.5：粗搜索（100×100 窗口，更快但粗糙）
          scale = 1.5：放大搜索（300×300 窗口，看更大范围）
        """
        windows = []
        # 按比例缩放
        scaled_w = int(self.win_w * scale)    # 如 200 * 0.5 = 100
        scaled_h = int(self.win_h * scale)    # 如 200 * 0.5 = 100
        stride_x = max(1, int(self.stride_x * scale))  # 至少 1px
        stride_y = max(1, int(self.stride_y * scale))

        idx = 0                               # 窗口编号（唯一标识符）
        y = 0
        # 垂直方向滑动
        while y + scaled_h <= self.pano_h:
            x = 0
            # 水平方向滑动
            while x + scaled_w <= self.pano_w:
                windows.append(ObservationWindow(
                    x=x, y=y,
                    width=scaled_w, height=scaled_h,
                    scale=scale, idx=idx
                ))
                idx += 1
                x += stride_x

            # 右边界对齐：如果最后一列窗口超出右边界，贴边再放一个
            if x < self.pano_w:
                windows.append(ObservationWindow(
                    x=self.pano_w - scaled_w, y=y,
                    width=scaled_w, height=scaled_h,
                    scale=scale, idx=idx
                ))
                idx += 1

            y += stride_y

        # 底边界对齐：如果最后一行超出下边界，贴边再放一排
        if y < self.pano_h:
            x = 0
            while x + scaled_w <= self.pano_w:
                windows.append(ObservationWindow(
                    x=x, y=self.pano_h - scaled_h,
                    width=scaled_w, height=scaled_h,
                    scale=scale, idx=idx
                ))
                idx += 1
                x += stride_x

        return windows

    def generate_all_scales(self) -> Dict[float, List[ObservationWindow]]:
        """
        生成所有尺度（粗+中+细）的窗口。

        返回:
          {0.5: [...粗窗口...], 1.0: [...标准窗口...], 1.5: [...放大窗口...], 2.0: [...最大窗口...]}
        """
        all_windows = {}
        all_windows[self.coarse_scale] = self.generate(self.coarse_scale)
        for s in self.fine_scales:
            all_windows[s] = self.generate(s)
        return all_windows

    @property
    def total_windows(self) -> int:
        """标准尺度下的候选窗口总数"""
        return len(self.generate(1.0))


# ============================================================================
# 工具函数（utility functions）
# ============================================================================

def load_image(path: str) -> np.ndarray:
    """
    从磁盘加载图像文件。

    【用途】
      当 PanoramaConfig.synthetic = False 时，加载真实全景图。

    【注意】
      OpenCV（cv2）默认以 BGR（蓝绿红）顺序读取，这里转为 RGB（红绿蓝）
      （matplotlib 等库用 RGB）。
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_panorama(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    将图像 resize（缩放）到目标尺寸。

    【用途】
      加载的真实图像尺寸不一，统一 resize 到配置的尺寸。
    """
    return cv2.resize(image, (target_width, target_height))
