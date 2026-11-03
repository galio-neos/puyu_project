"""
=============================================================================
real_detector.py — 真实目标检测器（YOLO26）
=============================================================================

【这个文件是干什么的】
  这是"真实世界"版的目标检测器：用深度学习模型 YOLO（You Only Look Once，
  即"你只看一次"，一种能实时检测物体的 AI 模型）来判断"这个观察窗口里
  有没有我们要找的目标"。

  在仿真实验（main.py）里，检测器是一个数学公式 SimulatedDetector——
  它"作弊"地知道目标的确切位置，按窗口离目标的距离算出置信度。
  而在真实场景（接摄像头）里没有这种已知信息，必须真的让 AI 模型去
  "看"图像内容，这就是本文件做的事。

【和仿真检测器的关系（关键）】
  两个检测器提供一模一样的接口：
      detect(window, scene) -> (置信度 confidence, 是否找到 found)
  所以搜索策略（main.py 里的 9 个类）一行都不用改，只要在运行时把
      searcher.detector = YOLODetector(...)
  就能把"假检测器"无缝换成"真检测器"。

【依赖】
  需要 ultralytics（YOLO 的官方 Python 库）和 torch（深度学习框架）。
  cv2（OpenCV）用来做图像通道转换。
=============================================================================
"""

import cv2                       # OpenCV：读图、颜色通道转换等图像操作
from ultralytics import YOLO     # ultralytics：YOLO 模型的官方 Python 库


# 中文目标名 → COCO 数据集里的英文类别名
# YOLO26 是在 COCO 数据集（80 类常见物体）上预训练的，类别名全是英文，
# 所以要先把用户输入的中文（如"水杯"）翻译成英文（cup），模型才认识。
CN_TO_COCO = {
    "水杯": "cup",      # cup = 杯子
    "汽车": "car",      # car = 汽车
    "椅子": "chair",    # chair = 椅子
    "电脑": "laptop",   # laptop = 笔记本电脑
    "书本": "book",     # book = 书
    "人": "person",     # person = 人
}


class YOLODetector:
    """
    用 YOLO26 模型做目标检测的"真实检测器"。

    【和仿真检测器 SimulatedDetector 的区别】
      SimulatedDetector.detect(window, scene)
        → 用数学公式算：窗口离目标越近置信度越高（因为知道目标在哪，是"开卷考试"）

      YOLODetector.detect(window, scene)
        → 真的把窗口图像喂给 AI 模型，模型看完内容后给出置信度（"闭卷考试"）
    """

    def __init__(self, target_category, model_path="models/weights/yolov8s.pt",
                 conf_threshold=0.50, imgsz=256):
        """
        初始化检测器。

        参数:
          target_category: 要找的目标类别（中文，如 "水杯"）
          model_path:      模型权重文件路径（.pt 文件，相当于"训练好的大脑"）
          conf_threshold:  置信度阈值——模型置信度超过它才认为"找到了"
          imgsz:           推理分辨率（喂给模型前，把图像缩放到这个尺寸）
        """
        # 加载 YOLO 模型。.pt 文件里存的是训练好的神经网络权重
        self.model = YOLO(model_path)

        # 把中文目标翻译成模型认识的英文类别名，存成列表。
        # 例如 target_category="水杯" → self.target_names = ["cup"]
        self.target_names = [CN_TO_COCO.get(target_category, target_category)]

        # 置信度阈值：模型觉得"这里就是目标"的概率要超过它才算"找到"。
        # YOLO 的真实置信度普遍偏低，所以默认 0.50（比仿真的 0.85 低很多）
        self.conf_threshold = conf_threshold

        # 推理分辨率：观察窗口只有 200×200，用 256 已经足够；
        # 默认的 640 会把小窗口放大后再推理，纯属浪费算力、拖慢速度
        self.imgsz = imgsz

    def detect(self, window, scene):
        """
        对指定窗口做一次真实检测，返回 (置信度, 是否找到)。

        【接口和 SimulatedDetector.detect() 完全一致】
        所以搜索策略在调用时，根本感觉不到"换了个检测器"。

        参数:
          window: 要观察的窗口（记录它在全景图里的位置和大小）
          scene:  全景场景（这里主要用 scene.image，即整张图像）

        返回:
          best:   这个窗口里"目标类别"的最高置信度 [0, 1]
          found:  best 是否超过了 conf_threshold（即"算不算找到"）
        """
        # 1. 从全景图里裁出这个窗口对应的图像小块
        crop = window.crop_from(scene.image)

        # 2. 颜色通道转换：项目内部统一用 RGB（红绿蓝），
        #    但 OpenCV/YOLO 用的是 BGR（蓝绿红），所以要转一下
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # 3. 把图像喂给模型做推理。
        #    verbose=False 表示不打印一堆日志；imgsz 指定推理分辨率
        results = self.model(crop_bgr, verbose=False, imgsz=self.imgsz)

        # 4. 从推理结果里找出"目标类别"的最高置信度
        best = 0.0
        for r in results:                     # r = 一张图的推理结果
            boxes = r.boxes                    # boxes = 模型检测到的所有候选框
            if boxes is None or len(boxes) == 0:
                continue                       # 没检测到任何东西，跳过

            # 逐个框看：cls 是类别编号，conf 是置信度，两者一一对应
            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                # 只关心"我们要找的那个类别"（如 cup），其它类别一律忽略
                if self.model.names[int(cls_id)] in self.target_names:
                    best = max(best, float(conf))   # 保留最高的那个置信度

        # 5. 返回（最高置信度, 是否超过阈值）
        return best, best >= self.conf_threshold
