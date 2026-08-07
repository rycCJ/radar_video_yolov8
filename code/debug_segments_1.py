import pandas as pd
import numpy as np
import json
from pathlib import Path
import pandas as pd
import numpy as np
import json
from pathlib import Path

def append_video_labels_to_radar(
    radar_csv_path, 
    radar_time_path, 
    video_events_path, 
    offset=0.0
):
    """
    向雷达特征CSV文件追加第6行：视频标签
    0: 静止, 1: Left, 2: Right
    """
    radar_csv_path = Path(radar_csv_path)
    
    print(f"--- 开始为雷达数据追加视频标签 ---")
    print(f"1. 读取雷达特征矩阵: {radar_csv_path.name}")
    try:
        # 读取原始数据 (假设是 5行N列)
        df_radar = pd.read_csv(radar_csv_path, header=None)
        
        # 检查形状，确保是 5xN
        if df_radar.shape[0] != 5:
            # 如果是 Nx5，转置一下
            if df_radar.shape[1] == 5:
                print("   [提示] 检测到 Nx5 格式，转置为 5xN")
                df_radar = df_radar.T
            else:
                print(f"   [警告] 数据格式异常 (行数={df_radar.shape[0]})，预期为 5 行。")
                # 依然继续，假设前5行是特征
                
        # 获取帧数 N
        num_frames = df_radar.shape[1]
        print(f"   雷达帧数: {num_frames}")
        
    except Exception as e:
        print(f"   [错误] 读取 CSV 失败: {e}")
        return

    # 2. 读取雷达时间戳
    print(f"2. 读取雷达时间戳: {Path(radar_time_path).name}")
    with open(radar_time_path, 'r') as f:
        radar_timestamps = np.array(json.load(f))
    
    # 截断以匹配矩阵列数
    radar_timestamps = radar_timestamps[:num_frames]

    # 3. 读取视频事件
    print(f"3. 读取视频标签: {Path(video_events_path).name}")
    df_video = pd.read_csv(video_events_path)

    # 4. 创建标签行 (第6行)
    # 初始化全为 0 (静止)
    video_label_row = np.zeros(num_frames, dtype=int)
    
    print(f"4. 进行时间对齐映射 (Offset = {offset}s)")
    
    matched_count = 0
    
    for _, event in df_video.iterrows():
        action = event['action']
        # 应用 offset: 视频时间 + offset = 雷达时间
        t_start = event['start_time'] + offset
        t_end = event['end_time'] + offset
        
        # 核心逻辑：找出所有时间落在 [t_start, t_end] 范围内的雷达帧索引
        # mask 是一个布尔数组，True 的位置表示该帧在挥手范围内
        mask = (radar_timestamps >= t_start) & (radar_timestamps <= t_end)
        
        # 映射标签值
        label_val = 0
        if action == 'Left':
            label_val = -1
        elif action == 'Right':
            label_val = 1
            
        # 赋值
        video_label_row[mask] = label_val
        
        if np.any(mask):
            matched_count += 1
            # print(f"   映射事件: {action} [{t_start:.2f}-{t_end:.2f}] -> 覆盖了 {np.sum(mask)} 帧")

    print(f"   成功映射了 {matched_count} 个视频片段到雷达帧。")

    # 5. 将新行追加到矩阵
    # 将 numpy 数组转为 DataFrame 的一行
    new_row_df = pd.DataFrame(video_label_row.reshape(1, -1))
    
    # 拼接到原数据下面 (变成 6行N列)
    df_combined = pd.concat([df_radar, new_row_df], ignore_index=True)
    
    # 6. 保存文件
    # 构造新文件名，避免覆盖原文件
    output_path = radar_csv_path.parent / f"{radar_csv_path.stem}_with_videolabel.csv"
    
    # 保存为不带表头的 CSV (保持 Matlab dlmwrite 格式)
    df_combined.to_csv(output_path, header=False, index=False)
    
    print(f"5. 结果已保存: {output_path.name}")
    print(f"   现在文件维度: {df_combined.shape} (预期为 6xN)")
    print("-" * 50)



def compare_radar_video_segments(
    radar_csv_path, 
    radar_time_path, 
    video_events_path, 
    output_path
):
    radar_csv_path = Path(radar_csv_path)
    
    # ==========================================
    # 1. 处理雷达数据 (复刻 Matlab 逻辑)
    # ==========================================
    print(f"1. 正在读取雷达特征: {radar_csv_path.name}")
    try:
        # 读取 5行N列 的矩阵 (没有表头)
        df_raw = pd.read_csv(radar_csv_path, header=None)
        
        # 确保是 5xN，如果是 Nx5 则转置
        if df_raw.shape[0] > 5 and df_raw.shape[1] == 5:
            print("   提示: 检测到 Nx5 格式，正在转置为 5xN...")
            df_raw = df_raw.T
            
        # 获取第 5 行 (Python 索引 4) 作为标签行
        # Matlab: ravp_matrix(5,:)
        radar_labels = df_raw.iloc[4, :].values
        
    except Exception as e:
        print(f"Error: 读取 CSV 失败 - {e}")
        return

    # 读取雷达时间戳
    print(f"2. 正在读取雷达时间戳: {Path(radar_time_path).name}")
    with open(radar_time_path, 'r') as f:
        radar_timestamps = np.array(json.load(f))
    
    # 确保长度对齐
    min_len = min(len(radar_labels), len(radar_timestamps))
    radar_labels = radar_labels[:min_len]
    radar_timestamps = radar_timestamps[:min_len]

    # --- 核心算法: 提取有效片段 ---
    # Matlab: nonzero_flag = ravp_matrix(5,:) ~= 0;
    nonzero_flag = (radar_labels != 0).astype(int)
    
    # Matlab: diff_flag = diff([0, nonzero_flag, 0]);
    # Python diff 长度会少1，所以前后加 0
    padded_flag = np.concatenate(([0], nonzero_flag, [0]))
    diff_flag = np.diff(padded_flag)
    
    # Matlab: block_start = find(diff_flag == 1);
    # Matlab find 返回的是 1-based索引，Python where 返回 0-based
    # 由于我们在前面补了0，这里的索引正好对应原数组的起始位置
    block_start = np.where(diff_flag == 1)[0]
    
    # Matlab: block_end = find(diff_flag == -1) - 1;
    # 这里的 -1 是因为 diff 导致索引偏移，且 Matlab find 是 1-based
    # 在 Python 中，diff_flag == -1 的位置其实是结束位置的"下一位"
    block_end = np.where(diff_flag == -1)[0] - 1
    
    print(f"   -> 原始检测到 {len(block_start)} 个雷达片段")

    # # --- 核心算法: 剔除首尾片段 ---
    # # Matlab: idxs_to_remove = unique([1, length(block_start)]);
    # if len(block_start) > 0:
    #     # Python 索引: 第一个是 0, 最后一个是 len-1
    #     # 如果只有一个片段，则首尾是同一个，都会被剔除
    #     idxs_to_remove = sorted(list(set([0, len(block_start) - 1])), reverse=True)
        
    #     print(f"   -> 正在自动剔除首尾片段 (Python索引): {idxs_to_remove}")
        
    #     # 必须倒序删除，否则索引会乱
    #     for k in idxs_to_remove:
    #         # 记录一下被删除的时间，方便调试
    #         del_start_t = radar_timestamps[block_start[k]]
    #         del_end_t = radar_timestamps[block_end[k]]
    #         print(f"      剔除: 索引{k} (时间 {del_start_t:.2f} - {del_end_t:.2f})")
            
    #         # 删除
    #         block_start = np.delete(block_start, k)
    #         block_end = np.delete(block_end, k)

    # 收集雷达片段信息
    comparison_data = []
    
    for i in range(len(block_start)):
        s_idx = block_start[i]
        e_idx = block_end[i]
        
        # 获取真实时间戳
        r_start_time = radar_timestamps[s_idx]
        r_end_time = radar_timestamps[e_idx]
        
        comparison_data.append({
            'Source': 'RADAR',
            'Event_ID': f'Radar_{i+1}',
            'Start_Time': r_start_time,
            'End_Time': r_end_time,
            'Duration': r_end_time - r_start_time,
            'Info': f'Frames: {s_idx}-{e_idx}'
        })

    # ==========================================
    # 2. 处理视频数据
    # ==========================================
    print(f"3. 正在读取视频事件: {Path(video_events_path).name}")
    if Path(video_events_path).exists():
        df_video = pd.read_csv(video_events_path)
        
        for idx, row in df_video.iterrows():
            # 假设视频时间戳是相对时间 (0.0s 开始)
            # 如果雷达时间戳是 Unix 时间戳 (17xxxxxx.xx)，这里会有巨大的差异
            # 你需要在 Excel 里肉眼看这个差异
            comparison_data.append({
                'Source': 'VIDEO',
                'Event_ID': f"Video_{idx+1}_{row['action']}",
                'Start_Time': row['start_time'],
                'End_Time': row['end_time'],
                'Duration': row['end_time'] - row['start_time'],
                'Info': f"Action: {row['action']}"
            })
    else:
        print("   警告: 未找到视频事件文件")

    # ==========================================
    # 3. 合并与保存
    # ==========================================
    if not comparison_data:
        print("没有找到任何数据片段。")
        return

    # 转为 DataFrame
    df_compare = pd.DataFrame(comparison_data)
    
    # 按【开始时间】排序，这样雷达和视频的对应事件就会排在一起
    df_compare = df_compare.sort_values(by='Start_Time')
    
    # 调整列顺序
    cols = ['Source', 'Event_ID', 'Start_Time', 'End_Time', 'Duration', 'Info']
    df_compare = df_compare[cols]
    
    # 保存
    print(f"4. 保存对比结果至: {output_path}")
    df_compare.to_csv(output_path, index=False)
    print("   完成！请打开 CSV 查看 Radar 和 Video 的时间差。")
    print("-" * 50)
    print("   [调试建议]")
    print("   1. 打开 CSV 文件。")
    print("   2. 找到相邻的 'RADAR' 和 'VIDEO' 行。")
    print("   3. 用 Excel 公式计算: Video_Start - Radar_Start。")
    print("   4. 这个差值的平均数，就是你需要填入 aligned_training.py 的 offset。")


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
        
        # (1) 雷达 CSV: [CASE]/feather_data/[CASE]_rvap_pos.csv
        radar_csv_path = case_folder / "feather_data" / f"{case_name}_rvap_pos.csv"
        
        # (2) 雷达时间戳: [CASE]/[CASE]_radar_timestamps.json
        radar_time_path = case_folder / f"{case_name}_radar_timestamps.json"
        
        # (3) 视频事件 CSV: [另一大文件夹]/[CASE]/wave_events.csv
        video_events_path = video_root / case_name / "wave_events.csv"

        output_path = case_folder / 'segment_comparison_check.csv'

        # --- 3. 检查文件是否存在 ---
        # 必须三个文件都存在才能处理
        if not radar_csv_path.exists():
            print(f"[跳过] 缺少雷达CSV: {case_name}")
            continue
        if not radar_time_path.exists():
            print(f"[跳过] 缺少时间戳: {case_name}")
            continue
        if not video_events_path.exists():
            print(f"[跳过] 缺少视频Events: {case_name} (在 thresh5_duration5 中未找到对应文件夹或文件)")
            continue

        # --- 4. 执行处理 ---
        print(f"[{i+1}/{len(case_folders)}] 处理: {case_name}")
        try:
            compare_radar_video_segments(
                radar_csv_path=radar_csv_path,
                radar_time_path=radar_time_path,
                video_events_path=video_events_path,
                output_path=output_path,
            )
            append_video_labels_to_radar(
                radar_csv_path=radar_csv_path,  
                radar_time_path=radar_time_path,
                video_events_path=video_events_path,
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
    # 请修改这里的路径
    # BASE_DIR = Path('/home/renyingcan/code/testhanshake/D5_handr8_lr_hd20_vd0_1300_renyingcan_20251217_171713')
    
    # # 1. Matlab 生成的特征文件 (5行N列)
    # RADAR_CSV = BASE_DIR / 'feather_data/D5_handr8_lr_hd20_vd0_1300_renyingcan_20251217_171713_rvap_pos.csv'
    
    # # 2. 雷达原始时间戳 JSON (必须有，用于获取真实时间)
    # RADAR_TIME = BASE_DIR / 'D5_handr8_lr_hd20_vd0_1300_renyingcan_20251217_171713_radar_timestamps.json'

 
    # # 3. YOLO 生成的视频事件 CSV
    # ANALY_VIDEO_DIR = Path('/home/renyingcan/code/thresh5_duration5/D5_handr8_lr_hd20_vd0_1300_renyingcan_20251217_171713')
    # VIDEO_CSV = ANALY_VIDEO_DIR / 'wave_events.csv'
    # # 4. 输出结果
    # OUTPUT_CSV = BASE_DIR / 'segment_comparison_check.csv'




    # compare_radar_video_segments(RADAR_CSV, RADAR_TIME, VIDEO_CSV, OUTPUT_CSV)

    # CALCULATED_OFFSET = 0.0  # <--- 请根据实际打印结果修改这个值
    
    # append_video_labels_to_radar(
    #     RADAR_CSV, 
    #     RADAR_TIME, 
    #     VIDEO_CSV, 
    #     offset=CALCULATED_OFFSET
    # )

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