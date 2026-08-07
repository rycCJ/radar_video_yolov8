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


class HandMotionTracker:
    def __init__(self, model_path, smoothing_window=5, movement_threshold=5):
        """
        初始化手部运动追踪器
        
        Args:
            model_path: YOLO模型路径
            smoothing_window: 平滑窗口大小（帧数）
            movement_threshold: 运动判定阈值（像素）
        """
        self.model = YOLO(model_path)
        self.smoothing_window = smoothing_window
        self.movement_threshold = movement_threshold
        
        # 存储左右手的历史位置
        self.left_hand_history = deque(maxlen=smoothing_window)
        self.right_hand_history = deque(maxlen=smoothing_window)
        
        # 存储检测结果
        self.tracking_data = {
            'left_hand': [],
            'right_hand': []
        }
        
        # 类别映射 (根据你的数据集)
        self.class_names = ['myleft', 'myright', 'yourleft', 'yourright']
        
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
    
    def process_frame(self, frame, frame_number, timestamp):
        """处理单帧"""
        results = self.model(frame, conf=0.25, verbose=False)
        
        left_hand_data = None
        right_hand_data = None
        
        # 提取检测结果
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                confidence = float(box.conf[0])
                
                center_x, center_y = self.get_hand_center(box)
                
                # 判断是左手还是右手（myleft或yourleft为左手）
                if 'left' in class_name.lower():
                    left_hand_data = {
                        'frame': frame_number,
                        'timestamp': timestamp,
                        'center_x': float(center_x),
                        'center_y': float(center_y),
                        'confidence': confidence,
                        'class': class_name
                    }
                elif 'right' in class_name.lower():
                    right_hand_data = {
                        'frame': frame_number,
                        'timestamp': timestamp,
                        'center_x': float(center_x),
                        'center_y': float(center_y),
                        'confidence': confidence,
                        'class': class_name
                    }
        
        # 更新历史记录并判断运动方向
        if left_hand_data:
            self.left_hand_history.append(left_hand_data)
            movement = self.determine_movement_direction(list(self.left_hand_history))
            left_hand_data['movement_direction'] = movement
            left_hand_data['movement_label'] = self.get_movement_label(movement)
            self.tracking_data['left_hand'].append(left_hand_data)
        
        if right_hand_data:
            self.right_hand_history.append(right_hand_data)
            movement = self.determine_movement_direction(list(self.right_hand_history))
            right_hand_data['movement_direction'] = movement
            right_hand_data['movement_label'] = self.get_movement_label(movement)
            self.tracking_data['right_hand'].append(right_hand_data)
        
        return left_hand_data, right_hand_data, results[0]
    
    def get_movement_label(self, movement):
        """获取运动标签文本"""
        if movement == 1:
            return "Left"
        elif movement == 0:
            return "Right"
        else:
            return "Still"
    
    def save_results(self, output_path):
        """保存追踪结果到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.tracking_data, f, indent=2, ensure_ascii=False)
        print(f"追踪数据已保存到: {output_path}")
    
    def save_to_csv(self, output_dir):
        """保存追踪结果到CSV文件"""
        output_dir = Path(output_dir)
        
        # 保存左手数据
        if self.tracking_data['left_hand']:
            left_df = pd.DataFrame(self.tracking_data['left_hand'])
            left_csv_path = output_dir / "left_hand_tracking.csv"
            left_df.to_csv(left_csv_path, index=False, encoding='utf-8')
            print(f"左手追踪数据已保存到: {left_csv_path}")
        
        # 保存右手数据
        if self.tracking_data['right_hand']:
            right_df = pd.DataFrame(self.tracking_data['right_hand'])
            right_csv_path = output_dir / "right_hand_tracking.csv"
            right_df.to_csv(right_csv_path, index=False, encoding='utf-8')
            print(f"右手追踪数据已保存到: {right_csv_path}")
        
        # 保存合并数据（左右手在同一个CSV中）
        combined_data = []
        
        # 获取所有帧号
        all_frames = set()
        for hand_data in self.tracking_data['left_hand']:
            all_frames.add(hand_data['frame'])
        for hand_data in self.tracking_data['right_hand']:
            all_frames.add(hand_data['frame'])
        
        # 创建帧号到数据的映射
        left_dict = {d['frame']: d for d in self.tracking_data['left_hand']}
        right_dict = {d['frame']: d for d in self.tracking_data['right_hand']}
        
        # 合并数据
        for frame in sorted(all_frames):
            row = {'frame': frame}
            
            if frame in left_dict:
                left_data = left_dict[frame]
                row['timestamp'] = left_data['timestamp']
                row['left_hand_center_x'] = left_data['center_x']
                row['left_hand_center_y'] = left_data['center_y']
                row['left_hand_confidence'] = left_data['confidence']
                row['left_hand_movement'] = left_data['movement_direction']
                row['left_hand_movement_label'] = left_data['movement_label']
                row['left_hand_class'] = left_data['class']
            else:
                row['timestamp'] = right_dict[frame]['timestamp'] if frame in right_dict else None
                row['left_hand_center_x'] = None
                row['left_hand_center_y'] = None
                row['left_hand_confidence'] = None
                row['left_hand_movement'] = None
                row['left_hand_movement_label'] = None
                row['left_hand_class'] = None
            
            if frame in right_dict:
                right_data = right_dict[frame]
                if 'timestamp' not in row or row['timestamp'] is None:
                    row['timestamp'] = right_data['timestamp']
                row['right_hand_center_x'] = right_data['center_x']
                row['right_hand_center_y'] = right_data['center_y']
                row['right_hand_confidence'] = right_data['confidence']
                row['right_hand_movement'] = right_data['movement_direction']
                row['right_hand_movement_label'] = right_data['movement_label']
                row['right_hand_class'] = right_data['class']
            else:
                row['right_hand_center_x'] = None
                row['right_hand_center_y'] = None
                row['right_hand_confidence'] = None
                row['right_hand_movement'] = None
                row['right_hand_movement_label'] = None
                row['right_hand_class'] = None
            
            combined_data.append(row)
        
        # 保存合并数据
        combined_df = pd.DataFrame(combined_data)
        combined_csv_path = output_dir / "combined_hand_tracking.csv"
        combined_df.to_csv(combined_csv_path, index=False, encoding='utf-8')
        print(f"合并追踪数据已保存到: {combined_csv_path}")
        
        return left_csv_path if self.tracking_data['left_hand'] else None, \
               right_csv_path if self.tracking_data['right_hand'] else None, \
               combined_csv_path
    
    def generate_report(self, output_dir):
        """生成分析报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("=" * 60)
        report.append("手部运动趋势分析报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for hand_type in ['left_hand', 'right_hand']:
            data = self.tracking_data[hand_type]
            if not data:
                continue
            
            report.append(f"\n{'左手' if hand_type == 'left_hand' else '右手'}统计:")
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
        
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        # 保存报告
        report_path = output_dir / "analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n分析报告已保存到: {report_path}")
    
    def plot_trajectory(self, output_path):
        """绘制手部运动轨迹图"""
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：SimHei（黑体）
        plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for idx, (hand_type, hand_name) in enumerate([('left_hand', '左手'), ('right_hand', '右手')]):
            data = self.tracking_data[hand_type]
            if not data:
                continue
            
            frames = [d['frame'] for d in data]
            x_coords = [d['center_x'] for d in data]
            y_coords = [d['center_y'] for d in data]
            movements = [d['movement_direction'] for d in data]
            
            # X坐标随时间变化
            ax1 = axes[idx, 0]
            colors = ['red' if m == 1 else 'blue' if m == 0 else 'gray' for m in movements]
            ax1.scatter(frames, x_coords, c=colors, alpha=0.6, s=10)
            ax1.set_xlabel('帧数')
            ax1.set_ylabel('X坐标 (像素)')
            ax1.set_title(f'{hand_name} X坐标轨迹\n(红=向左, 蓝=向右, 灰=静止)')
            ax1.grid(True, alpha=0.3)
            
            # 2D轨迹图
            ax2 = axes[idx, 1]
            ax2.scatter(x_coords, y_coords, c=colors, alpha=0.6, s=10)
            ax2.plot(x_coords, y_coords, 'k-', alpha=0.2, linewidth=0.5)
            ax2.set_xlabel('X坐标 (像素)')
            ax2.set_ylabel('Y坐标 (像素)')
            ax2.set_title(f'{hand_name} 2D运动轨迹')
            ax2.invert_yaxis()  # 图像坐标系Y轴向下
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"运动轨迹图已保存到: {output_path}")
        plt.close()


def process_video(video_path, model_path, output_dir, show_video=False):
    """
    处理视频并分析手部运动
    
    Args:
        video_path: 输入视频路径
        model_path: YOLO模型路径
        output_dir: 输出目录
        show_video: 是否显示实时视频
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化追踪器
    tracker = HandMotionTracker(model_path, smoothing_window=5, movement_threshold=5)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {total_frames/fps:.2f} 秒")
    
    # 创建输出视频
    output_video_path = output_dir / "annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    frame_number = 0
    
    print("\n开始处理视频...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_number / fps
        
        # 处理当前帧
        left_data, right_data, results = tracker.process_frame(frame, frame_number, timestamp)
        
        # 在帧上绘制检测结果
        annotated_frame = results.plot()
        
        # 添加运动信息文本
        y_offset = 30
        if left_data:
            movement_text = f"Left Hand: ({left_data['center_x']:.0f}, {left_data['center_y']:.0f}) - {left_data['movement_label']}"
            color = (0, 0, 255) if left_data['movement_direction'] == 1 else (255, 0, 0) if left_data['movement_direction'] == 0 else (128, 128, 128)
            cv2.putText(annotated_frame, movement_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        if right_data:
            movement_text = f"Right Hand: ({right_data['center_x']:.0f}, {right_data['center_y']:.0f}) - {right_data['movement_label']}"
            color = (0, 0, 255) if right_data['movement_direction'] == 1 else (255, 0, 0) if right_data['movement_direction'] == 0 else (128, 128, 128)
            cv2.putText(annotated_frame, movement_text, (10, y_offset), 
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
                    print("用户中断处理")
                    break
            except cv2.error as e:
                if frame_number == 0:  # 只在第一帧显示警告
                    print("警告: 无法显示视频窗口（无GUI支持），将继续处理但不显示实时画面")
                    print("提示: 可以将 show_video 参数设置为 False")
                show_video = False  # 禁用后续的显示尝试
        
        # 显示进度
        if frame_number % 30 == 0:
            progress = (frame_number / total_frames) * 100
            print(f"处理进度: {progress:.1f}% ({frame_number}/{total_frames})")
        
        frame_number += 1
    
    # 释放资源
    cap.release()
    out.release()
    if show_video:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    print(f"\n视频处理完成!")
    print(f"标注视频已保存到: {output_video_path}")
    
    # 保存追踪数据
    json_path = output_dir / "tracking_data.json"
    tracker.save_results(json_path)
    
    # 保存CSV数据
    print("\n保存CSV数据...")
    tracker.save_to_csv(output_dir)
    
    # 生成报告
    tracker.generate_report(output_dir)
    
    # 绘制轨迹图
    plot_path = output_dir / "trajectory_plot.png"
    tracker.plot_trajectory(plot_path)
    
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == '__main__':
    # 配置参数
    VIDEO_PATH = '/home/renyingcan/code/handshake/D5_handl17_lr_hd30_vdn15_1300_renyingcan_20251217_172902/D5_handl17_lr_hd30_vdn15_1300_renyingcan_20251217_172902.mp4'  # 修改为你的视频路径
    MODEL_PATH = '/home/renyingcan/code/handcode/runs/detect/hand_detection2/weights/best.pt'  # 修改为你的模型路径
    OUTPUT_DIR = '/home/renyingcan/code/hand_tracking_result'
    SHOW_VIDEO = True  # 是否显示实时处理视频
    
    # 处理视频
    process_video(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR, show_video=SHOW_VIDEO)