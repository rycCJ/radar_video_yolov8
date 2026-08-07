import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import json
matplotlib.use('Agg')  # 服务器端绘图不弹窗

def align_features(radar_csv_path,radar_time_path, video_events_path, offset=0.0, save=False):
    radar_csv_path = Path(radar_csv_path)
    
    # ==========================================
    # 1. 读取 Matlab 生成的 CSV
    # ==========================================
    print(f"正在读取雷达特征文件: {radar_csv_path.name} ...")
    
    # Matlab dlmwrite 默认没有表头，且通常是 5行N列
    try:
        # header=None 表示没有列名
        df_temp = pd.read_csv(radar_csv_path, header=None)
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # 检查形状，如果是 (5, N)，我们需要转置成 (N, 5)
    if df_temp.shape[0] == 5 and df_temp.shape[1] > 5:
        print("  -> 检测到矩阵为 5行N列，正在转置...")
        df_radar = df_temp.T
        df_radar.columns = ['Range', 'Velocity', 'Angle', 'Power', 'Matlab_Label']
    else:
        # 假设已经是 N行5列
        df_radar = df_temp
        df_radar.columns = ['Range', 'Velocity', 'Angle', 'Power', 'Matlab_Label']

    # ==========================================
    # 2. 生成雷达时间轴
    # ==========================================
    # 根据 Matlab 代码：T_frame = 0.05
    try:
        # --- 修改开始：使用 json 读取列表格式 [0.0, 0.046...] ---
        with open(radar_time_path, 'r') as f:
            time_data = json.load(f) # 直接解析列表
            
        # 确保读出来的是列表
        if not isinstance(time_data, list):
            print("  [错误] 时间戳文件内容不是列表格式！")
            return
            
        # 简单的数据一致性检查
        if len(time_data) != len(df_radar):
            print(f"  [警告] 时间戳行数 ({len(time_data)}) 与 特征数据行数 ({len(df_radar)}) 不一致！")
       
        # 赋值时间戳
        df_radar['timestamp'] = time_data
        # --- 修改结束 ---
        
        print(f"  -> 雷达数据共 {len(df_radar)} 帧，起止时间: {df_radar['timestamp'].iloc[0]:.3f}s ~ {df_radar['timestamp'].iloc[-1]:.3f}s")
        
    except Exception as e:
        print(f"读取时间戳文件失败: {e}")
        print("  提示：请检查时间戳文件是否为标准的 JSON 列表格式 [0.1, 0.2, ...]")
        return

    # ==========================================
    # 3. 读取 YOLO 挥手事件
    # ==========================================
    df_events = pd.read_csv(video_events_path)
    print(f"  -> 读取到 {len(df_events)} 个视频挥手事件")

    # ==========================================
    # 4. 对齐打标 (应用 Offset)
    # ==========================================
    # 初始化标签列
    df_radar['YOLO_Label'] = 'Still'
    df_radar['YOLO_Label_ID'] = 0  # 0:Still, 1:Left, 2:Right
    
    print(f"  -> 应用 Offset: {offset} 秒")

    for _, event in df_events.iterrows():
        action = event['action']
        # 视频时间 + 偏移 = 雷达时间
        t_start = event['start_time'] + offset
        t_end = event['end_time'] + offset
        
        # 找到对应时间段的雷达帧
        mask = (df_radar['timestamp'] >= t_start) & (df_radar['timestamp'] <= t_end)
        
        df_radar.loc[mask, 'YOLO_Label'] = action
        
        if action == 'Left':
            df_radar.loc[mask, 'YOLO_Label_ID'] = 1
        elif action == 'Right':
            df_radar.loc[mask, 'YOLO_Label_ID'] = 2

    # ==========================================
    # 5. 可视化验证 (画速度图 vs 标签)
    # ==========================================
    plt.figure(figsize=(16, 8))
    
    # --- 子图 1: 速度 (Velocity) ---
    # 速度是最能体现挥手方向的特征
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(df_radar['timestamp'] , df_radar['Velocity'], color='blue', label='Radar Velocity (m/s)')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.ylabel("Velocity (m/s)")
    plt.title(f"Alignment Check (Offset={offset}s) - Velocity")
    plt.grid(True, alpha=0.3)
    
    # 画 YOLO 标签背景
    # Left = Red, Right = Green
    left_mask = df_radar['YOLO_Label_ID'] == 1
    right_mask = df_radar['YOLO_Label_ID'] == 2
    
    # 使用 y轴的范围来填充背景
    ylim = ax1.get_ylim()
    plt.fill_between(df_radar['timestamp'], ylim[0], ylim[1], where=left_mask, 
                     color='red', alpha=0.3, label='Video: Left')
    plt.fill_between(df_radar['timestamp'], ylim[0], ylim[1], where=right_mask, 
                     color='green', alpha=0.3, label='Video: Right')
    plt.legend(loc='upper right')

    # --- 子图 2: 能量 (Power/SNR) ---
    # 能量可以辅助判断动作发生的时刻
    plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(df_radar['timestamp'], df_radar['Power'], color='purple', label='Radar Power/SNR')
    plt.ylabel("Power (dB)")
    plt.xlabel("Time (seconds)")
    plt.title("Radar Power")
    plt.grid(True, alpha=0.3)
    
    # 同样画上背景
    ylim2 = plt.gca().get_ylim()
    plt.fill_between(df_radar['timestamp'], ylim2[0], ylim2[1], where=left_mask, 
                     color='red', alpha=0.3)
    plt.fill_between(df_radar['timestamp'], ylim2[0], ylim2[1], where=right_mask, 
                     color='green', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    img_name = f"check_alignment_{radar_csv_path.stem}_offset_{offset}.png"  # test/stem/featherdata/
    save_img_path = radar_csv_path.parent / img_name
    plt.savefig(save_img_path)
    print(f"🖼️  对齐检查图已保存: {save_img_path}")
    plt.close()

    # ==========================================
    # 6. 保存最终训练数据
    # ==========================================
    if save:
        # 保存文件名: 原文件名_aligned.csv
        save_csv_path = radar_csv_path.parent / f"{radar_csv_path.stem}_aligned.csv"
        
        # 只保存你需要的列，比如: [Range, Velocity, Angle, Power, YOLO_Label, YOLO_Label_ID]
        df_radar.to_csv(save_csv_path, index=False)
        print(f"✅ [成功] 已保存对齐后的数据: {save_csv_path}")
    else:
        print("⚠️ [预览模式] 未保存 CSV。请检查图片确认对齐后，设置 save=True")
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
            align_features(
                radar_csv_path=radar_csv_path,
                radar_time_path=radar_time_path,
                video_events_path=video_events_path,
                offset=offset,
                save=save
            )
            success_count += 1
            print("  -> 完成")
        except Exception as e:
            print(f"  [ERROR] 处理出错: {e}")
            # traceback.print_exc() # 如果需要看详细报错，取消注释

    print("=" * 80)
    print(f"批量处理结束。成功: {success_count}/{len(case_folders)}")

if __name__ == "__main__":
    # --- 配置路径 ---
    # 1. 雷达数据根目录 (包含许多 D5_... 文件夹)
    RADAR_ROOT_DIR = '/home/renyingcan/code/testhanshake'
    # 2. 视频/YOLO分析结果根目录 (包含 wave_events.csv 的上级)
    VIDEO_ROOT_DIR = '/home/renyingcan/code/thresh5_duration5'

    # 3. 调整 Offset (单位: 秒)
    # 正数: 视频标签向右移 (雷达比视频早)
    # 负数: 视频标签向左移 (雷达比视频晚) 
    MY_OFFSET = 0       #雷达-视频 

    # 先 save=False 看图，调整 Offset，准了再改成 True
    # 是否保存图片/结果 (建议先 False 跑一遍看 log，没问题再改 True)
    SAVE_RESULT = False

    batch_process_all_cases(
    radar_root=RADAR_ROOT_DIR,
    video_root=VIDEO_ROOT_DIR,
    offset=MY_OFFSET,
    save=SAVE_RESULT
)
    


#     BASE_DIR = Path('/home/renyingcan/code/testhanshake/D5_handr8_lr_hd20_vd0_1300_renyingcan_20251217_171713')
#     RADAR_CSV = BASE_DIR / 'feather_data/D5_handr8_lr_hd20_vd0_1300_renyingcan_20251217_171713_rvap_pos.csv'
#     ANALY_VIDEO_DIR = Path('/home/renyingcan/code/thresh5_duration5/D5_handr8_lr_hd20_vd0_1300_renyingcan_20251217_171713')

#     VIDEO_EVENTS  = ANALY_VIDEO_DIR / 'wave_events.csv'

#     # 时间戳文件 (包含 [0.0, 0.046...] 的文件)
#     RADAR_TIME_CSV = BASE_DIR / 'D5_handr8_lr_hd20_vd0_1300_renyingcan_20251217_171713_radar_timestamps.json' 

#     align_features(RADAR_CSV, RADAR_TIME_CSV, VIDEO_EVENTS, offset=MY_OFFSET, save=False)
