import cv2
import numpy as np
from ultralytics import YOLO
import json
import csv
import pandas as pd
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys


# annotated_video.mp4：画了框、写了移动方向的视频（可视化结果）。
# tracking_data.json：原始数据，给程序员看的。
# hand_tracking.csv：表格数据，可以用 Excel 打开，每一行是每一帧的坐标。
# analysis_report.txt：总结报告（例如：“这个视频里，手向左动了50帧，向右动了30帧”）。
# trajectory_plot.png：一张图表，画出了手在空中的运动轨迹。


plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 指定默认字体：SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题
class HandMotionTracker:
    def __init__(self, model_path, smoothing_window=5, movement_threshold=20, single_class=True):
        """
        初始化手部运动追踪器
        
        Args:
            model_path: YOLO模型路径
            smoothing_window: 平滑窗口大小（帧数）
            movement_threshold: 运动判定阈值（像素）
            single_class: 是否为单类别模型（只检测手）
        """
        self.model = YOLO(model_path)
        self.smoothing_window = smoothing_window
        self.movement_threshold = movement_threshold
        self.single_class = single_class
        
        # 存储手的历史位置
        self.hand_history = deque(maxlen=smoothing_window)
        
        # 存储检测结果
        self.tracking_data = {
            'hand': []
        }
        
        # 类别映射
        if single_class:
            self.class_names = ['hand']
        else:
            self.class_names = ['myleft', 'myright', 'yourleft', 'yourright']
        
    def reset(self):
        """重置追踪器状态，用于处理新视频"""
        self.hand_history.clear()
        self.tracking_data = {
            'hand': []
        }
    
    def get_hand_center(self, box):
        """获取检测框的中心坐标"""
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y
    
    def determine_movement_direction(self, history):
        """
        判断运动方向
        
        Returns:
            1: 向左移动
            0: 向右移动
            None: 静止或无法判断
        """
        if len(history) < 2:
            return None
        
        # 计算平均移动方向
        movements = []
        for i in range(1, len(history)):
            dx = history[i]['center_x'] - history[i-1]['center_x']
            if abs(dx) > self.movement_threshold:
                movements.append(dx)
        
        if not movements:
            return None  # 静止
        
        avg_movement = np.mean(movements)
        
        # 向左移动（x坐标减小）返回1，向右移动（x坐标增加）返回0
        if avg_movement < -self.movement_threshold:
            return 1  # 向左
        elif avg_movement > self.movement_threshold:
            return 0  # 向右
        else:
            return None  # 静止
    
    def get_movement_label(self, movement):
        """获取运动标签文本"""
        if movement == 1:
            return "Left"
        elif movement == 0:
            return "Right"
        else:
            return "Still"
    
    def process_frame(self, frame, frame_number, timestamp):
        """处理单帧，只保留置信度最高的检测结果"""
        results = self.model(frame, conf=0.25, verbose=False)
        
        hand_data = None
        
        # 提取检测结果，只保留置信度最高的
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # 找到置信度最高的检测框
            max_conf_idx = 0
            max_conf = 0
            
            for i, box in enumerate(boxes):
                confidence = float(box.conf[0])
                if confidence > max_conf:
                    max_conf = confidence
                    max_conf_idx = i
            
            # 只处理置信度最高的检测框
            box = boxes[max_conf_idx]
            cls_id = int(box.cls[0])
            class_name = self.class_names[cls_id]
            confidence = float(box.conf[0])
            
            center_x, center_y = self.get_hand_center(box)
            
            hand_data = {
                'frame': frame_number,
                'timestamp': timestamp,
                'center_x': float(center_x),
                'center_y': float(center_y),
                'confidence': confidence,
                'class': class_name
            }
        
        # 更新历史记录并判断运动方向
        if hand_data:
            self.hand_history.append(hand_data)
            movement = self.determine_movement_direction(list(self.hand_history))
            hand_data['movement_direction'] = movement
            hand_data['movement_label'] = self.get_movement_label(movement)
            self.tracking_data['hand'].append(hand_data)
        
        return hand_data, results[0]
    
    def save_results(self, output_path):
        """保存追踪结果到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.tracking_data, f, indent=2, ensure_ascii=False)
    
    def save_to_csv(self, output_dir):
        """保存追踪结果到CSV文件"""
        output_dir = Path(output_dir)
        
        # 保存手部数据
        if self.tracking_data['hand']:
            hand_df = pd.DataFrame(self.tracking_data['hand'])
            hand_csv_path = output_dir / "hand_tracking.csv"
            hand_df.to_csv(hand_csv_path, index=False, encoding='utf-8')
            print(f"  ✓ CSV数据: {hand_csv_path.name}")
    
    def generate_report(self, output_dir):
        """生成分析报告"""
        output_dir = Path(output_dir)
        
        report = []
        report.append("=" * 60)
        report.append("手部运动趋势分析报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        data = self.tracking_data['hand']
        if data:
            report.append(f"手部统计:")
            report.append(f"  总检测帧数: {len(data)}")
            
            # 统计运动方向
            left_count = sum(1 for d in data if d['movement_direction'] == 1)
            right_count = sum(1 for d in data if d['movement_direction'] == 0)
            still_count = sum(1 for d in data if d['movement_direction'] is None)
            
            report.append(f"  向左移动帧数: {left_count}")
            report.append(f"  向右移动帧数: {right_count}")
            report.append(f"  静止帧数: {still_count}")
            
            # 平均置信度
            avg_conf = np.mean([d['confidence'] for d in data])
            report.append(f"  平均检测置信度: {avg_conf:.3f}")
        else:
            report.append("未检测到手部数据")
        
        report_text = "\n".join(report)
        
        # 保存报告
        report_path = output_dir / "analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    def plot_trajectory(self, output_path):
        """绘制手部运动轨迹图"""

        data = self.tracking_data['hand']
        if not data:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        frames = [d['frame'] for d in data]
        x_coords = [d['center_x'] for d in data]
        y_coords = [d['center_y'] for d in data]
        movements = [d['movement_direction'] for d in data]
        
        # X坐标随时间变化
        ax1 = axes[0]
        colors = ['red' if m == 1 else 'blue' if m == 0 else 'gray' for m in movements]
        ax1.scatter(frames, x_coords, c=colors, alpha=0.6, s=10)
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('X (Pixels)')
        ax1.set_title('X motion trajectory\n(red=Left, blue=Right, gray=Still)')
        ax1.grid(True, alpha=0.3)
        
        # 2D轨迹图
        ax2 = axes[1]
        ax2.scatter(x_coords, y_coords, c=colors, alpha=0.6, s=10)
        ax2.plot(x_coords, y_coords, 'k-', alpha=0.2, linewidth=0.5)
        ax2.set_xlabel('X (Pixels)')
        ax2.set_ylabel('Y (Pixels)')
        ax2.set_title('2D motion trajectory')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def process_single_video(video_path, model_path, output_dir, tracker, show_video=False):
    """
    处理单个视频
    
    Args:
        video_path: 输入视频路径
        model_path: YOLO模型路径
        output_dir: 输出目录
        tracker: HandMotionTracker实例
        show_video: 是否显示实时视频
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 重置追踪器
    tracker.reset()
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  错误: 无法打开视频 {video_path}")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  视频信息: {width}x{height}, {fps:.2f} FPS, {total_frames} 帧, {total_frames/fps:.2f} 秒")
    
    # 创建输出视频
    output_video_path = output_dir / "annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_number / fps
        
        # 处理当前帧
        hand_data, results = tracker.process_frame(frame, frame_number, timestamp)
        
        # 在帧上绘制检测结果
        annotated_frame = results.plot()
        
        # 添加运动信息文本
        if hand_data:
            movement_text = f"Hand: ({hand_data['center_x']:.0f}, {hand_data['center_y']:.0f}) - {hand_data['movement_label']} (conf: {hand_data['confidence']:.2f})"
            color = (0, 0, 255) if hand_data['movement_direction'] == 1 else (255, 0, 0) if hand_data['movement_direction'] == 0 else (128, 128, 128)
            cv2.putText(annotated_frame, movement_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 添加时间戳
        cv2.putText(annotated_frame, f"Time: {timestamp:.2f}s | Frame: {frame_number}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 写入输出视频
        out.write(annotated_frame)
        
        # 显示视频（仅在支持GUI的环境中）
        if show_video:
            try:
                cv2.imshow('Hand Motion Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("  用户中断处理")
                    break
            except cv2.error:
                if frame_number == 0:
                    print("  警告: 无法显示视频窗口（无GUI支持）")
                show_video = False
        
        frame_number += 1
    
    # 释放资源
    cap.release()
    out.release()
    if show_video:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    print(f"  ✓ 标注视频: {output_video_path.name}")
    
    # 保存追踪数据
    json_path = output_dir / "tracking_data.json"
    tracker.save_results(json_path)
    print(f"  ✓ JSON数据: {json_path.name}")
    
    # 保存CSV数据
    tracker.save_to_csv(output_dir)
    
    # 生成报告
    tracker.generate_report(output_dir)
    print(f"  ✓ 分析报告: analysis_report.txt")
    
    # 绘制轨迹图
    plot_path = output_dir / "trajectory_plot.png"
    tracker.plot_trajectory(plot_path)
    print(f"  ✓ 轨迹图: {plot_path.name}")
    
    return True


def batch_process_videos(base_dir, model_path, output_base_dir, show_video=False):
    """
    批量处理所有视频
    
    Args:
        base_dir: 包含所有视频文件夹的基础目录
        model_path: YOLO模型路径
        output_base_dir: 输出基础目录
        show_video: 是否显示实时视频
    """
    base_dir = Path(base_dir)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建追踪器（只创建一次，重复使用）
    print("正在加载YOLO模型...")
    tracker = HandMotionTracker(
        model_path=model_path, 
        smoothing_window=5, 
        movement_threshold=5,
        single_class=True  # 使用单类别模型
    )
    print("模型加载完成!\n")
    
    # 查找所有视频文件
    video_files = []
    for folder in base_dir.iterdir():
        if folder.is_dir():
            # 在每个文件夹中查找.mp4文件
            for video_file in folder.glob("*.mp4"):
                video_files.append(video_file)
    
    if not video_files:
        print(f"错误: 在 {base_dir} 中未找到任何视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件\n")
    print("=" * 80)
    
    # 统计信息
    success_count = 0
    failed_count = 0
    failed_videos = []
    
    # 处理每个视频
    for idx, video_path in enumerate(video_files, 1):
        # 获取视频文件名（不含扩展名）作为输出文件夹名
        video_name = video_path.stem
        output_dir = output_base_dir / video_name
        
        print(f"\n[{idx}/{len(video_files)}] 处理视频: {video_path.name}")
        print(f"  输出目录: {output_dir}")
        
        try:
            success = process_single_video(
                video_path=video_path,
                model_path=model_path,
                output_dir=output_dir,
                tracker=tracker,
                show_video=show_video
            )
            
            if success:
                success_count += 1
                print(f"  ✓ 完成!")
            else:
                failed_count += 1
                failed_videos.append(video_path.name)
                print(f"  ✗ 失败!")
                
        except Exception as e:
            failed_count += 1
            failed_videos.append(video_path.name)
            print(f"  ✗ 错误: {str(e)}")
        
        print("-" * 80)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("批量处理完成!")
    print("=" * 80)
    print(f"总计: {len(video_files)} 个视频")
    print(f"成功: {success_count} 个")
    print(f"失败: {failed_count} 个")
    
    if failed_videos:
        print("\n失败的视频:")
        for video_name in failed_videos:
            print(f"  - {video_name}")
    
    print(f"\n所有结果已保存到: {output_base_dir}")


if __name__ == '__main__':
    # 配置参数
    BASE_DIR = '/home/renyingcan/code/testhanshake'  # 包含所有视频文件夹的基础目录
    MODEL_PATH = 'runs/detect/hand_detection_robust4/weights/best.pt'  # 单类别模型路径 /home/renyingcan/code/handcode/runs/detect/hand_detection_robust4/weights/best.pt
    OUTPUT_BASE_DIR = '/home/renyingcan/code/thresh5_duration5_batch'  # 输出基础目录
    SHOW_VIDEO = False  # 是否显示实时视频（服务器环境建议False）
    
    # 批量处理视频
    batch_process_videos(BASE_DIR, MODEL_PATH, OUTPUT_BASE_DIR, show_video=SHOW_VIDEO)
