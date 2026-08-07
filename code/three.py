import pandas as pd
import numpy as np
import json
from pathlib import Path
import pandas as pd
import numpy as np
import json
from pathlib import Path


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
# --- 运行配置 ---
if __name__ == "__main__":


    # 1. 雷达数据根目录 (包含许多 D5_... 文件夹)
    RADAR_ROOT_DIR = '/home/renyingcan/code/hand_shake'
    # 2. 视频/YOLO分析结果根目录 (包含 wave_events.csv 的上级)
    VIDEO_ROOT_DIR = '/home/renyingcan/code/test_thresh5_duration5'

    CALCULATED_OFFSET = 0.0
    SAVE_RESULT = False

    batch_process_all_cases(radar_root = RADAR_ROOT_DIR,
                            video_root = VIDEO_ROOT_DIR,
                            
                            offset=CALCULATED_OFFSET,
                            save=SAVE_RESULT)