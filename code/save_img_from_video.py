# 简单的抽帧脚本D5_handr2_lr_hd10_vd0_1300_liguoqiang_20251217_150044
import cv2
import os

# 1. 定义视频路径
video_path = '/home/renyingcan/code/testhanshake/D5_handr14_lr_hd30_vd0_1300_liuhangcheng_20251217_163932/D5_handr14_lr_hd30_vd0_1300_liuhangcheng_20251217_163932.mp4'

# 2. 定义保存目录 (你原本想存的地方)
output_dir = '/home/renyingcan/code/handdata_single/new_data/images'

# --- 核心修改：如果文件夹不存在，强制创建它 ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"文件夹不存在，已自动创建: {output_dir}")
else:
    print(f"文件夹已存在: {output_dir}")

# 3. 读取视频
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开 (防止路径写错导致没有任何反应)
if not cap.isOpened():
    print(f"错误: 无法打开视频文件，请检查路径是否正确：\n{video_path}")
    exit()

count = 0
saved_count = 0

print("开始抽帧...")
while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    if count % 5 == 0: # 每5帧存一张
        # 拼接完整的文件名
        save_path = os.path.join(output_dir, f'bad_video_{count}.jpg')
        
        # 保存图片
        success = cv2.imwrite(save_path, frame)
        
        if success:
            saved_count += 1
            if saved_count % 10 == 0:
                print(f"已保存 {saved_count} 张图片...")
        else:
            print(f"保存失败: {save_path}")
            
    count += 1

cap.release()
print(f"全部完成！共保存了 {saved_count} 张图片到: {output_dir}")