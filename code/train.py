from ultralytics import YOLO



# 加载模型：请来了一位有基础的学生（YOLOv11）。
# 准备数据：你准备了一堆手的照片和对应的答案（数据集）。
# 训练 (train)：让学生闭关刷题100遍，直到能准确把手圈出来。
# 保存结果：把学成归来的学生大脑备份下来（best.pt）。
# 推理 (predict)：拿几张新照片测试一下，看它能不能在上面准确地画出框。

# 加载预训练的YOLO11模型
model = YOLO('yolo11l.pt')  # 可选: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
# model_path = '/home/renyingcan/code/handcode/runs/detect/hand_detection4/weights/best.pt'
# model = YOLO(model_path)
# 训练模型
results = model.train(
    data='/home/renyingcan/code/handdata_single/data.yaml',
    epochs=500,              # 训练轮数
    imgsz=640,               # 图像大小
    batch=16,                # 批次大小，根据GPU内存调整
    name='hand_detection_robust',   # 实验名称
    patience=50,             # 早停耐心值
    save=True,               # 保存检查点
    device=0,                # 使用GPU 0，如果是CPU则设为'cpu'
    workers=8,               # 数据加载器工作进程数
    project='runs/detect',   # 项目文件夹

    # --- 核心修改：增强抗干扰能力 ---
    
    # 1. 旋转增强 (重要)
    # 挥手时手腕角度变化大，让图片随机旋转 +/- 45度
    degrees=15.0,     

    # 2. 缩放增强 (重要)
    # 手前手后大小不一，增加缩放范围 (+/- 70%)
    scale=0.7,        

    # 3. 剪切与透视 (模拟摄像头角度不正)
    shear=2.0,       # 剪切
    perspective=0.0005, # 透视变化

    # 4. 颜色与亮度 (模拟光照变化)
    hsv_h=0.015,     # 色调微调
    hsv_s=0.7,       # 饱和度大幅波动 (有些视频色彩淡，有些浓)
    hsv_v=0.4,       # 亮度波动 (模拟阴影)

    # 5. 翻转
    fliplr=0.5,      # 50%概率左右翻转 (左手变右手)
    
    # 6. 马赛克增强 (YOLO的核心，必开)
    mosaic=1.0,      # 100% 开启，把4张图拼一起，让模型学习小目标和部分遮挡
    
    # 7. Mixup (混合)
    mixup=0.1,       # 10%概率把两张图叠在一起，模拟残影或重叠
    
    # 8. 关键：模糊处理 (如果YOLO版本支持直接传参)
    # 注意：ultralytics 标准参数里对 blur 的支持是自动的或者通过 albumentations
    # 但通过 mosaic 和 scale 已经能模拟很多模糊情况
    # 这里的 copy_paste 也有助于增加样本密度
    copy_paste=0.1, 
    close_mosaic=10,        # 最后10个epoch关闭mosaic
)

# 训练完成后验证
metrics = model.val()

# 测试模型
test_results = model.predict(
    source='/home/renyingcan/code/handdata_single/test/images',
    save=True,
    conf=0.25  # 置信度阈值
)

print("训练完成！")
print(f"最佳模型保存在: runs/detect/hand_detection/weights/best.pt")