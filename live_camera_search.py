"""
=============================================================================
live_camera_search.py — 外部摄像头 + YOLO26 实时主动目标搜索
=============================================================================

【这是做什么的】
  把项目从"合成场景仿真 / 静态真实图"拓展到【真实摄像头】：
    - 从外部摄像头（或视频文件、图片）逐帧取画面；
    - 每一帧当作一张"全景场景"，用【和仿真完全相同的搜索策略】在上面
      主动搜索目标——观察窗口的顺序决策逻辑一行都不改；
    - 检测器换成 YOLO26（默认 COCO 预训练 yolo26n.pt，真实场景更准；
      也可切到训练好的 synth_yolo26_v4/best.pt）。

【为什么"检测顺序策略不变"】
  8 个策略 + Combined 全部继承 BaseSearcher，统一接口 search(scene, grid)。
  本文件只做两件事：
    1. 把场景来源从"合成生成器"换成"摄像头这一帧"；
    2. 把 searcher.detector 从模拟检测器换成 YOLODetector（一行赋值）。
  其余决策逻辑（B1 顺序 / B2 随机 / B3 显著性 / B4 置信度 / I1~I4 / Combined）
  原封不动。

【运行方式】（在项目根目录 d:/forth-project 下执行）
  python code/live_camera_search.py                    # 默认摄像头 source=0，找"水杯"
  python code/live_camera_search.py --target 人        # 换目标
  python code/live_camera_search.py --strategy ior     # 换策略
  python code/live_camera_search.py --source data/real/5594.jpg_wh860.jpg  # 无摄像头用图片测试
  python code/live_camera_search.py --model outputs/runs/detect/runs/synth_yolo26_v4/weights/best.pt

【按键】
  q / ESC   退出
  r         重搜当前帧（不取新帧）
  空格 / n  取下一帧
  1~9       实时切换策略（1=B1 ... 9=Combined）

【注意】
  稳态约 50~150 ms/帧（约 8~20 fps），已接近实时；首帧含模型加载约 0.5~1 s。
  提速关键：imgsz=256——观察窗口仅 200×200，默认 640 会把小窗口放大推理、纯浪费算力。
=============================================================================
"""

# 标准库：argparse=解析命令行参数，os=处理文件路径
import argparse
import os

# 第三方库：cv2=OpenCV（摄像头/图像），numpy=数值计算（多维数组）
import cv2
import numpy as np

# 本项目其它模块（都在同一个 code 文件夹下）：
from config import WindowConfig          # 观察窗口大小/步长等配置
from environment import PanoramaScene, WindowGrid, load_image  # 场景、窗口网格、读图
from real_detector import YOLODetector, CN_TO_COCO            # 真实 YOLO 检测器、中英映射
from main import (
    SequentialScanSearcher,
    RandomScanSearcher,
    SaliencyGuidedSearcher,
    DetectionConfidenceSearcher,
    InhibitionOfReturnSearcher,
    ScenePriorGuidedSearcher,
    CoarseToFineSearcher,
    VisualMemorySearcher,
    CombinedSearcher,
)

# 项目根目录：相对路径一律按它解析，保证 VSCode 点 ▶ 运行时 cwd 不在根目录也找得到
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 默认模型：COCO 预训练 YOLO26（真实场景更准）；训练好的合成模型作为可切换项。
DEFAULT_MODEL = os.path.join(ROOT, "models", "weights", "yolo26n.pt")
TRAINED_MODEL = os.path.join(ROOT, "outputs", "runs", "detect", "runs",
                             "synth_yolo26_v4", "weights", "best.pt")

# ====================================================================
# 傻瓜式配置区：在 VSCode 里改这里、直接点 ▶ 运行，不用开终端。
# 终端运行时这些是默认值，也能用 --xxx 覆盖。
# ====================================================================
SOURCE = "1"              # 摄像头索引（0=默认摄像头）；或图片/视频路径（无摄像头测试）
TARGET = "水杯"            # 中文目标：水杯/汽车/椅子/电脑/书本/人
STRATEGY = "combined"     # b1/b2/b3/b4/ior/scene/coarse/memory/combined
MAX_STEPS = 25            # 搜索预算（观察窗口上限）
STOP_THRESH = 0.50        # 停止阈值（YOLO 置信度偏低，0.50）
CONF_THRESH = 0.25        # 判定"找到"的置信度下限
MAX_WIDTH = 960           # 画面最长边（等比缩放）
IMGSZ = 256               # YOLO 推理分辨率（窗口 200×200，256 够用）

WINDOW_TITLE = "Live Active Search (YOLO26)"

# 策略注册表：按键顺序即 1~9 切换顺序。
STRATEGY_CHOICES = [
    ("b1",       SequentialScanSearcher,       "B1 Sequential"),
    ("b2",       RandomScanSearcher,           "B2 Random"),
    ("b3",       SaliencyGuidedSearcher,       "B3 Saliency"),
    ("b4",       DetectionConfidenceSearcher,  "B4 Confidence"),
    ("ior",      InhibitionOfReturnSearcher,   "I1 IOR"),
    ("scene",    ScenePriorGuidedSearcher,     "I2 ScenePrior"),
    ("coarse",   CoarseToFineSearcher,         "I3 CoarseToFine"),
    ("memory",   VisualMemorySearcher,         "I4 VisualMemory"),
    ("combined", CombinedSearcher,             "Combined"),
]
# 从上面的注册表整理出两个字典/列表，方便按名字快速查到策略类：
#   STRATEGIES["ior"] -> InhibitionOfReturnSearcher 这个类
#   STRATEGY_KEYS      -> ["b1","b2","b3","b4","ior","scene","coarse","memory","combined"]
STRATEGIES = {key: cls for key, cls, _ in STRATEGY_CHOICES}
STRATEGY_KEYS = [key for key, _, _ in STRATEGY_CHOICES]

# 常见的图片文件后缀。用来判断 --source 给的是"图片"还是"视频/摄像头"
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# ============================================================================
# 工具函数
# ============================================================================

def _looks_like_image(path: str) -> bool:
    """判断一个来源是不是"单张图片"（而不是摄像头索引或视频文件）。"""
    return isinstance(path, str) and path.lower().endswith(_IMG_EXTS)


def resize_frame(img: np.ndarray, max_width: int) -> np.ndarray:
    """等比缩放到 max_width 宽（保持宽高比，避免 1600×600 强制拉伸导致畸变）。"""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (max_width, int(round(h * scale))))


def _detect_boxes(model, crop_bgr, target_name, conf_thresh=0.25):
    """
    对单个窗口裁剪图做检测，返回【局部坐标】下的目标框列表。

    返回: [(x1, y1, x2, y2, conf), ...]   （相对窗口左上角）
    与 real_search_merge.detect_local_boxes() 同一套写法：检测器只给置信度、
    这里额外把边界框取出来，用于在画面上框出"找到的目标"。
    """
    results = model(crop_bgr, verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for cls_id, conf, xyxy in zip(r.boxes.cls.tolist(),
                                      r.boxes.conf.tolist(),
                                      r.boxes.xyxy.tolist()):
            if model.names[int(cls_id)] == target_name and conf >= conf_thresh:
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                boxes.append((x1, y1, x2, y2, float(conf)))
    return boxes


# ============================================================================
# 核心：在一帧上跑一次主动搜索（策略原封不动）
# ============================================================================

def run_search(frame_rgb, target_category, args, strat_idx, detector):
    """把一帧画面当场景，运行选定策略，返回 (TrialResult, PanoramaScene)。"""
    H, W = frame_rgb.shape[:2]

    scene = PanoramaScene(
        image=frame_rgb,
        scene_labels=np.zeros((H, W), dtype=int),   # 真实图无语义分割
        scene_label_map={},                          # 空 → 场景先验自动退化为常量
        target_positions=[],                         # 真实图无真值
        target_category=target_category,
    )
    grid = WindowGrid(W, H, WindowConfig())

    cls = STRATEGY_CHOICES[strat_idx][1]
    searcher = cls(max_steps=args.max_steps)
    searcher.detector = detector                      # ⭐ 换真实 YOLO 检测器
    searcher._stop_threshold = args.stop_thresh       # ⭐ 覆盖停止阈值（适配 YOLO）

    result = searcher.search(scene, grid)
    return result, scene


# ============================================================================
# 渲染：画观察窗口轨迹 + 找到框 + 状态栏
# ============================================================================

def render(frame_bgr, scene, result, detector, target_name, strategy_label):
    img = frame_bgr.copy()
    obs = result.observations
    n = max(1, len(obs))

    # ---- 1. 按观察顺序画每个窗口（蓝→绿渐变 + 步号）----
    for i, o in enumerate(obs):
        w = o.window
        ratio = i / n
        color = (int(255 * ratio), 170, int(255 * (1 - ratio)))
        cv2.rectangle(img, (w.x, w.y), (w.x + w.width, w.y + w.height), color, 1)
        cv2.putText(img, str(o.step), (w.x + 3, w.y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # ---- 2. 高亮最后一步（找到=红，未找到=橙）----
    if obs:
        last = obs[-1].window
        color = (0, 0, 255) if result.found_target else (0, 165, 255)
        cv2.rectangle(img, (last.x, last.y),
                      (last.x + last.width, last.y + last.height), color, 2)

        # 找到时：在最后窗口上取目标框，映射到全局坐标画绿框
        crop = last.crop_from(scene.image)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        for (lx1, ly1, lx2, ly2, conf) in _detect_boxes(
                detector.model, crop_bgr, target_name, conf_thresh=0.25):
            gx1, gy1 = last.x + lx1, last.y + ly1
            gx2, gy2 = last.x + lx2, last.y + ly2
            cv2.rectangle(img, (int(gx1), int(gy1)), (int(gx2), int(gy2)),
                          (0, 255, 0), 2)
            cv2.putText(img, f"{target_name} {conf:.2f}",
                        (int(gx1), int(gy1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---- 3. 顶部状态栏 ----
    max_conf = max(result.confidences) if result.confidences else 0.0
    status = (f"{strategy_label} | target={target_name} | "
              f"steps={result.total_steps} | found={result.found_target} | "
              f"max_conf={max_conf:.2f}")
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (40, 40, 40), -1)
    cv2.putText(img, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)

    # ---- 4. 底部按键提示 ----
    hint = "q/ESC=quit  r=re-search  space/n=next  1-9=switch strategy"
    cv2.putText(img, hint, (8, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


# ============================================================================
# 主流程
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="外部摄像头 + YOLO26 实时主动目标搜索")
    p.add_argument("--source", default=SOURCE,
                   help="摄像头索引(0/1/...)、视频文件路径、或图片路径（无摄像头测试）")
    p.add_argument("--target", default=TARGET)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="YOLO 权重（默认 COCO 预训练 yolo26n.pt；可切训练好的 best.pt）")
    p.add_argument("--strategy", default=STRATEGY, choices=STRATEGY_KEYS,
                   help="搜索策略：" + "/".join(STRATEGY_KEYS))
    p.add_argument("--max-steps", type=int, default=MAX_STEPS,
                   help="搜索预算（观察窗口上限）")
    p.add_argument("--stop-thresh", type=float, default=STOP_THRESH,
                   help="停止搜索的置信度阈值（YOLO 置信度偏低，默认 0.50）")
    p.add_argument("--conf-thresh", type=float, default=CONF_THRESH,
                   help="判定'找到目标'的置信度下限（默认 0.25）")
    p.add_argument("--max-width", type=int, default=MAX_WIDTH,
                   help="画面最长边（等比缩放，保宽高比）")
    p.add_argument("--imgsz", type=int, default=IMGSZ,
                   help="YOLO 推理分辨率（窗口 200×200，256 够用，640 更慢）")
    args = p.parse_args()

    # 解析 source：数字 → 摄像头索引；其余 → 图片/视频路径
    # 相对路径按项目根目录解析，保证 VSCode 点 ▶ 运行时也能找到
    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    elif isinstance(src, str) and not os.path.isabs(src):
        src = os.path.join(ROOT, src)
    is_live = isinstance(src, int) or not _looks_like_image(src)

    target_name = CN_TO_COCO.get(args.target, args.target)
    detector = YOLODetector(args.target, model_path=args.model,
                            conf_threshold=args.conf_thresh, imgsz=args.imgsz)
    # 预热：首帧推理含模型加载，约 0.5~1s；提前跑一次避免第一帧卡顿
    detector.model(np.zeros((200, 200, 3), np.uint8), imgsz=args.imgsz, verbose=False)
    strat_idx = STRATEGY_KEYS.index(args.strategy)

    cap = None
    fixed_rgb = None
    if is_live:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头/视频源: {src}")
        print(f"已打开视频源: {src}，目标 = {args.target} ({target_name})，"
              f"模型 = {args.model}")
    else:
        fixed_rgb = load_image(src)
        print(f"已加载图片: {src}，目标 = {args.target} ({target_name})，"
              f"模型 = {args.model}")

    print("按键: q/ESC=退出  r=重搜  空格/n=下一帧  1-9=切换策略")
    print("-" * 60)

    current_rgb = None
    need_frame = True
    while True:
        # ---- 取帧 ----
        if cap is not None:
            if need_frame:
                ok, bgr = cap.read()
                if not ok:
                    break
                bgr = resize_frame(bgr, args.max_width)
                current_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            if current_rgb is None:
                current_rgb = resize_frame(fixed_rgb, args.max_width)
        need_frame = True

        # ---- 搜索 ----
        result, scene = run_search(current_rgb, args.target, args, strat_idx, detector)

        # ---- 控制台摘要 ----
        max_conf = max(result.confidences) if result.confidences else 0.0
        print(f"[{STRATEGY_CHOICES[strat_idx][2]}] found={result.found_target} "
              f"steps={result.total_steps} max_conf={max_conf:.2f}")

        # ---- 渲染 + 显示 ----
        disp = render(cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR),
                      scene, result, detector, target_name,
                      STRATEGY_CHOICES[strat_idx][2])
        cv2.imshow(WINDOW_TITLE, disp)
        key = cv2.waitKey(30) & 0xFF

        # ---- 按键 ----
        if key in (ord("q"), 27):            # q / ESC
            break
        elif key == ord("r"):                # 重搜当前帧
            need_frame = False
        elif key in (ord(" "), ord("n")):    # 下一帧
            need_frame = True
        elif ord("1") <= key <= ord("9"):    # 切换策略
            strat_idx = key - ord("1")
            need_frame = False
            print(f"切换策略 -> {STRATEGY_CHOICES[strat_idx][2]}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("\n已退出。")


if __name__ == "__main__":
    main()
