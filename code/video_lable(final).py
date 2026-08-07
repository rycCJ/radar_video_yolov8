import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from tqdm import tqdm  # 进度条库
import matplotlib.pyplot as plt
import os
import json

# 在你的 process_video 函数中，找到调用 _analyze_motion 的那一行，修改参数
# 还是很抖  →  继续增大 velocity_thresh (试到 8.0 或 10.0)。
# 真正的挥手检测不到了  →  减小 min_duration (改为 3 或 4)。

class HandWaveAnalyzer:
    def __init__(self, model_path, conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def process_video(self, video_path, output_dir):
        """
        处理单个视频的主函数
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"正在分析: {video_path.name} ...")
        json_path = video_path.parent / f"{video_path.stem}_video_timestamps.json"
        if json_path.exists():
            print(f"  [信息] 找到时间戳文件: {json_path.name}")
        else:
            print(f"  [警告] 未找到时间戳JSON，将退回到使用FPS计算时间（可能导致对齐误差）")
            json_path = None


        # --- 第一步：收集原始数据 (YOLO检测) ---
        raw_df, fps, width, height = self._collect_raw_data(video_path,json_path)
        
        if raw_df.empty:
            print(f"  [警告] 视频 {video_path.name} 未检测到任何手部数据。")
            return
            
        # --- 第二步：计算与分割 (平滑 + 状态判断) ---
        # smooth_window: 平滑窗口，越大越平滑但会有延迟，建议15-31之间
        # velocity_thresh: 像素变化阈值，超过这个值才算动
        # analyzed_df, segments = self._analyze_motion(raw_df, smooth_window=15, velocity_thresh=3.0)
        analyzed_df, segments = self._analyze_motion(
            raw_df, 
            smooth_window=15,      # 增大平滑窗口 (建议 21 或 31)
            velocity_thresh=5.0,   # 增大速度阈值 (建议 5.0 - 10.0)
            min_duration=5         # 新增参数：动作必须持续 6 帧(约0.2秒)才算数
        )
        # --- 第三步：保存数据报表 ---
        # 保存挥手时间段 (用于对齐雷达)
        pd.DataFrame(segments).to_csv(output_dir / 'wave_events.csv', index=False)
        # 保存逐帧数据 (用于调试)
        analyzed_df.to_csv(output_dir / 'frame_analysis.csv', index=False)
        # 绘制图表
        self._plot_charts(analyzed_df, segments, output_dir / 'analysis_plot.png')
        
        # --- 第四步：生成可视化视频 (根据分析结果绘制) ---
        self._render_result_video(video_path, output_dir / 'result_vis.mp4', analyzed_df, fps, width, height)
        
        print(f"  [完成] 结果已保存在: {output_dir}")

    def _collect_raw_data(self, video_path,json_path = None):
        """第一遍扫描：只负责收集坐标"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- 修改了这里：加载真实时间戳 ---
        real_timestamps = []
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    real_timestamps = json.load(f)
                # 简单校验长度
                if abs(len(real_timestamps) - total_frames) > 10:
                    print(f"  [注意] JSON时间戳数量 ({len(real_timestamps)}) 与 视频帧数 ({total_frames}) 差异较大，请注意！")
            except Exception as e:
                print(f"  [错误] 读取JSON失败: {e}")
        
        # --- 字典用于存储所有检测到的ID的数据 ---

        data_list = []
        frame_idx = 0
        
        # 使用tqdm显示进度条

        pbar = tqdm(total=total_frames, desc="  Step 1/2: YOLO检测", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
                        
            # YOLO 推理
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            current_time = 0.0
            if frame_idx < len(real_timestamps):
                # 使用 JSON 中的真实时间 (通常是 Unix 时间戳)
                current_time = real_timestamps[frame_idx]
            else:
                # 降级方案：如果没有JSON或溢出，用FPS计算
                print("没有JSON或溢出")
                current_time = frame_idx / fps
            row = {
                'frame': frame_idx,
                'timestamp': current_time,
                'raw_x': np.nan,
                'raw_y': np.nan,
                'conf': 0.0
            }
            
            if len(results[0].boxes) > 0:
                # 取置信度最高的框
                best_box = max(results[0].boxes, key=lambda x: x.conf[0])
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                row['raw_x'] = (x1 + x2) / 2
                row['raw_y'] = (y1 + y2) / 2
                row['conf'] = float(best_box.conf[0])
                
            data_list.append(row)
            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        return pd.DataFrame(data_list), fps, width, height

    def _analyze_motion(self, df, smooth_window=15, velocity_thresh=5.0, min_duration=5):
        """
        核心算法：平滑与逻辑判断 (增强防抖版)
        :param velocity_thresh: 建议提高到 5.0 或 8.0 (根据视频分辨率调整)
        :param min_duration: 最小持续帧数，小于这个长度的动作会被忽略
        """
        # 1. 插值补全缺失值
        df['x_smooth'] = df['raw_x'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        # fillna(method='bfill')：向后填充; fillna(method='ffill')：向前填充

        # 2. Savgol滤波平滑 (增大窗口可以抑制高频抖动)
        # 确保窗口是奇数且不超过数据长度
        if len(df) > smooth_window:
            window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            df['x_smooth'] = savgol_filter(df['x_smooth'], window_length=window, polyorder=2)
        # polyorder=2：使用二次多项式拟合
        # 3. 计算速度
        df['velocity'] = df['x_smooth'].diff().fillna(0)
        
        # 4. 初步判定状态 (Raw State)
        # 这里提高门槛，只有速度绝对值很大才算动
        def get_raw_state(v):
            if v > velocity_thresh: return 'Right'
            if v < -velocity_thresh: return 'Left'
            return 'Still'
            
        df['raw_state'] = df['velocity'].apply(get_raw_state)
        
        # 5. 关键修改：短时噪声过滤 (Debouncing)
        # 逻辑：如果一段动作持续时间小于 min_duration 帧，强制置为 Still
        
        # 利用 shift 找到状态变化的时刻
        # group_id 只有在状态变化时才会 +1
        df['group_id'] = (df['raw_state'] != df['raw_state'].shift()).cumsum()
        
        # 统计每一组的长度
        group_counts = df.groupby('group_id')['raw_state'].transform('count')
        
        # 修正状态：如果该组长度太短，且不是静止状态，则强制视为静止 (噪声)
        # 意思就是：如果你只往左动了 2 帧，我就当你没动
        df['state'] = np.where(
            (group_counts < min_duration) & (df['raw_state'] != 'Still'), 
            'Still', 
            df['raw_state']
        )

        # 6. 提取连续的时间段 (Segments) - 基于修正后的 state
        segments = []
        if len(df) == 0: return df, segments

        # 重新分组 (因为上面修改了 state，可能会合并一些碎片)
        df['final_group'] = (df['state'] != df['state'].shift()).cumsum()
        
        grouped = df.groupby('final_group')
        for _, group in grouped:
            state_type = group['state'].iloc[0]
            
            # 我们只关心 Left 和 Right，Still 不记录进事件表
            if state_type == 'Still': continue 
            
            segments.append({
                'action': state_type,
                'start_time': group['timestamp'].min(),
                'end_time': group['timestamp'].max(),
                'start_frame': group['frame'].min(),
                'end_frame': group['frame'].max(),
                'avg_velocity': group['velocity'].mean()
            })
            
        return df, segments

    def _render_result_video(self, video_path, save_path, analyzed_df, fps, width, height):
        """第二遍扫描：根据分析好的数据画图"""
        cap = cv2.VideoCapture(str(video_path))
        out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frame_idx = 0
        total_frames = len(analyzed_df)
        pbar = tqdm(total=total_frames, desc="  Step 2/2: 生成视频", unit="frame")
        
        # 颜色定义 (BGR)
        colors = {
            'Right': (255, 0, 0),   # 蓝色
            'Left': (0, 0, 255),    # 红色
            'Still': (128, 128, 128) # 灰色
        }
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx < len(analyzed_df):
                # 获取这一帧的分析结果
                row = analyzed_df.iloc[frame_idx]
                state = row['state']
                x_pos = row['x_smooth'] # 使用平滑后的坐标画圈
                y_pos = row['raw_y'] if not np.isnan(row['raw_y']) else height/2
                
                # 1. 在画面右上角写大字状态
                color = colors.get(state, (255, 255, 255))
                cv2.putText(frame, f"Action: {state}", (30, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                # 2. 画出平滑后的手部中心点
                if not np.isnan(x_pos):
                    cv2.circle(frame, (int(x_pos), int(y_pos)), 10, color, -1)
                
                # 3. 显示时间戳
                cv2.putText(frame, f"Time: {row['timestamp']:.2f}s", (30, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        out.release()

    def _plot_charts(self, df, segments, save_path):
        """绘制可视化图表"""
        plt.figure(figsize=(12, 6))
        
        # 画X轴轨迹
        plt.plot(df['timestamp'], df['x_smooth'], 'k-', alpha=0.5, label='Smoothed X')
        
        # 标记区间
        for seg in segments:
            color = 'blue' if seg['action'] == 'Right' else 'red'
            plt.axvspan(seg['start_time'], seg['end_time'], color=color, alpha=0.3)
            # 在区间上写字
            mid_t = (seg['start_time'] + seg['end_time']) / 2
            plt.text(mid_t, df['x_smooth'].max(), seg['action'], ha='center', fontsize=8, color=color)
            
        plt.title('Hand Movement Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (px)')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def batch_process_folders(input_root, output_root, model_path):
    """
    递归遍历大文件夹，处理所有视频，并保持目录结构
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    # 初始化分析器
    analyzer = HandWaveAnalyzer(model_path)
    
    # 查找所有MP4文件 (递归查找)
    video_files = list(input_root.rglob("*.mp4"))
    
    if not video_files:
        print("未找到任何 .mp4 文件！")
        return

    print(f"共找到 {len(video_files)} 个视频文件，准备开始处理...")
    print("="*60)

    for i, video_file in enumerate(video_files):
        # 1. 计算相对路径
        # 例如: input/user1/gesture1/vid.mp4 -> relative: user1/gesture1/vid.mp4
        relative_path = video_file.relative_to(input_root)
        
        # 2. 构建输出文件夹路径
        # output/user1/gesture1/vid (我们为每个视频单独建一个文件夹放结果)
        # .stem 是文件名去后缀 (vid)
        save_folder = output_root / relative_path.parent
        
        print(f"\n[{i+1}/{len(video_files)}] 处理: {relative_path}")
        
        try:
            analyzer.process_video(video_file, save_folder)
        except Exception as e:
            print(f"  [ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()


def generate_timestamp_alignment_csv(
    radar_time_path, 
    video_time_path,
    video_events_path, 
    output_path, 
    offset=0.0
):
    """
    生成对齐CSV：包含雷达时间戳、对应的视频时间戳、以及视频标签。
    解决 20Hz(雷达) 与 30Hz(视频) 的不一致。
    """
    print(f"--- 正在生成时间戳对齐文件 ---")

    # 1. 读取雷达原始时间戳 (20Hz)
    # 假设是从 json 读取的 list/array
    print(f"读取雷达视频时间: {Path(radar_time_path).name}")
    with open(radar_time_path, 'r') as f:
        radar_timestamps = np.array(json.load(f))
    with open(video_time_path, 'r') as f:
        video_timestamps = np.array(json.load(f))
    
    # 2. 读取视频事件定义 (Action, Start_time, End_time)
    print(f"读取视频事件: {Path(video_events_path).name}")
    df_video = pd.read_csv(video_events_path)

    # 3. 准备存储对齐数据的列表
    alignment_results = []

    # 核心逻辑：遍历每一个雷达时间点，计算它在视频坐标系下的位置
    for r_time in radar_timestamps:
        # 换算逻辑：雷达时间 - offset = 视频相对时间
        # 这样即使帧率不同，我们也能知道这一帧雷达发生时，视频播放到了第几秒
        v_time_ref = r_time - offset
        # --- 解决帧率不齐：寻找视频中最接近 target_v_time 的那一帧 ---
        # 计算视频时间轴中所有点与目标时间的差绝对值
        distances = np.abs(video_timestamps - v_time_ref)
        closest_idx = np.argmin(distances)
        matched_v_time = video_timestamps[closest_idx] # 找到的最匹配的视频时间戳

        # 默认标签
        label = 0
        action_name = "Static"

        # 在视频事件表中查找，看当前的 v_time_ref 落在哪个动作区间
        # 这样就自动处理了“丢弃视频帧”或“拉开距离”的问题，因为我们只看雷达点
        match = df_video[
            (df_video['start_time'] <= matched_v_time) & 
            (df_video['end_time'] >= matched_v_time)
        ]

        if not match.empty:
            action_name = match.iloc[0]['action']
            if action_name == 'Left':
                label = -1
            elif action_name == 'Right':
                label = 1
        
        # 将这一帧的对照关系存入
        alignment_results.append({
            'radar_timestamp': r_time,        # 原始雷达时间 (20Hz)
            'matched_video_timestamp': matched_v_time,
            # 'video_equivalent_time': v_time_ref, # 对应视频的时刻
            'video_label': label,             # 映射后的标签
            'action': action_name             # 动作名称(方便核对)
        })

    # # 4. 转换为 DataFrame 并导出
    # 创建后转置：.T 让它变成 4行N列    
    df_horizontal = pd.DataFrame(alignment_results).T    
    # 导出时包含所有字段，方便你在 Excel 里肉眼观察对齐效果
    df_horizontal.to_csv(output_path, index=False)
    
    print(f"对齐完成！")
    print(f"总行数: {len(df_horizontal)} (与雷达帧数一致)")
    print(f"保存路径: {output_path}")


def batch_process_all_cases(radar_root, video_root, offset=0.0, save=False):
    """
    批量遍历 radar_root 下的所有子文件夹，匹配 video_root 下的对应文件，并执行 align_features
    """
    radar_root = Path(radar_root)
    video_root = Path(video_root)

    # 1. 获取 radar_root 下所有的子目录 (即每个 case 的文件夹)
    # 例如: /home/.../testhanshake/D5_handr8_...
    case_folders = [p for p in radar_root.iterdir() if p.is_dir()]
    
    # 按名称排序，保证处理顺序一致
    case_folders.sort()

    print(f"在 {radar_root} 下共找到 {len(case_folders)} 个目录，开始匹配处理...")
    print("=" * 80)

    success_count = 0

    for i, case_folder in enumerate(case_folders):
        # 获取文件夹名称，例如: "D5_handr8_lr_hd20_vd0_1300_renyingcan_20251217_171713"
        case_name = case_folder.name
        
        # --- 2. 构建目标文件路径 ---
        
        # # (1) 雷达 CSV: [CASE]/feather_data/[CASE]_rvap_pos.csv
        # radar_csv_path = case_folder / "feather_data" / f"{case_name}_rvap_pos.csv"
        
        # (2) 雷达时间戳: [CASE]/[CASE]_radar_timestamps.json
        radar_time_path = case_folder / f"{case_name}_radar_timestamps.json"
        video_time_path = case_folder / f"{case_name}_video_timestamps.json"
        
        
        # (3) 视频事件 CSV: [另一大文件夹]/[CASE]/wave_events.csv
        video_events_path = video_root / case_name / "wave_events.csv"

        output_path = case_folder / 'segment_comparison_check.csv'

        # --- 3. 检查文件是否存在 ---

        if not radar_time_path.exists():
            print(f"[跳过] 缺少时间戳: {case_name}")
            continue
        if not video_time_path.exists():
            print(f"[跳过] 缺少视频时间戳: {case_name}")
            continue
        if not video_events_path.exists():
            print(f"[跳过] 缺少视频Events: {case_name} (在 thresh5_duration5 中未找到对应文件夹或文件)")
            continue

        # --- 4. 执行处理 ---
        print(f"[{i+1}/{len(case_folders)}] 处理: {case_name}")
        try:
            generate_timestamp_alignment_csv(
                radar_time_path=radar_time_path,
                video_time_path=video_time_path,
                video_events_path=video_events_path,
                output_path=output_path,
                offset=offset,
            )

            success_count += 1
            print("  -> 完成")
        except Exception as e:
            print(f"  [ERROR] 处理出错: {e}")
            # traceback.print_exc() # 如果需要看详细报错，取消注释

    print("=" * 80)
    print(f"批量处理结束。成功: {success_count}/{len(case_folders)}")


if __name__ == "__main__":
    # --- 配置区域 ---
    
    # 1. 大文件夹路径 (输入)
    INPUT_ROOT = '/home/renyingcan/code/hand_shake' 
    
    # 2. 结果保存的大文件夹路径 (输出)
    OUTPUT_ROOT = '/home/renyingcan/code/test_thresh5_duration5'

    
    # 3. 模型路径
    MODEL_PATH = 'runs/detect/hand_detection_robust4/weights/best.pt'
    
    CALCULATED_OFFSET = 0.0
    SAVE_RESULT = False
    batch_process_folders(INPUT_ROOT, OUTPUT_ROOT, MODEL_PATH)



    batch_process_all_cases(radar_root = INPUT_ROOT,
                            video_root = OUTPUT_ROOT,
                            offset=CALCULATED_OFFSET,
                            save=SAVE_RESULT)