CONDA yolov8

该路径下所有的hand开头的文件 都是和本项目有关的
tracking_result 结果
code 代码
    batch_process.py 批量化操作处理视频数据
    train.py 训练模型
    transfer_label 把data改为 data_single
    权重文件路径
    /home/weipengfei/code/handcode/runs/detect/hand_detectionX
data 原始训练数据 4个类别 myleft myright yourleft yourright
data_single 修改后的训练数据 1个类别 hand
handshake 你们提供的视频数据文件



debug_segments_1.py 



一只手：
batch_process 生成视频annotated_video视频 带框 有点晃
ren_process  生成results_video视频，点，比较稳定
ren_process_2 点 ，生成的视频点不跟随




**requirements.txt** 为运行代码所需要的库，运行之前，直接 `pip install requirements.txt`
**video_lable(final).py**为最终标注文件，在最后的`if __name__ == "__main__":`中：
- MODEL_PATH为best.pt模型文件路径
- INPUT_ROOT为hand_shake挥手大文件夹的目录
- OUTPUT_ROOT为中间文件生成目录，可以自己选择

如果处理之后的视频肉眼可见的标签不对，对于参数调整：
- 在 process_video 函数中，找到调用 _analyze_motion 的那一行，修改参数
- 还是很抖  →  继续增大 velocity_thresh (试到 8.0 或 10.0)。
- 真正的挥手检测不到了  →  减小 min_duration (改为 3 或 4)。

**aligned_training.py**为绘图文件
此文件需要用到matlab生成得feather_data文件夹
可以生成雷达和视频的融合的图片 可以看offset
**debug_segments_1.py**
生成csv文件，需要用到matlab生成得feather_data文件夹
可以在雷达的rvap中添加视频标签matlab生成（D5_handr14_lr_hd30_vd0_1300_liuhangcheng_20251217_163932_rvap_pos_with_videolabel.csv）  
生成segment_comparison_check.csv,里面有雷达和视频挥手时间段和方向，可看offset

**ren_process.py**
决定视频中挥手片段的范围，通过调节：thread  和duration，且生成wave_events.csv和简图

**code/handcode/save_img_from_video.py** 
将视频裁成图片

**process_finish**
ren_process运行生成的文件。（thresh5_duration5）

**hand_finish**
testhanshake 原始数据处理之后，feather data里面多出了文件


