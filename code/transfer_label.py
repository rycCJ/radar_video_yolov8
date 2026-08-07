import os
from pathlib import Path

def convert_labels_to_single_class(dataset_path):
    """将多类别标注转换为单类别（所有类别->0）"""
    
    dataset_path = Path(dataset_path)
    
    for split in ['train', 'valid', 'test']:
        labels_dir = dataset_path / split / 'labels'
        
        if not labels_dir.exists():
            print(f"警告: {labels_dir} 不存在")
            continue
            
        print(f"处理 {split} 数据集...")
        
        label_files = list(labels_dir.glob('*.txt'))
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # 将所有类别ID改为0
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # 确保是有效的YOLO格式
                    # 格式: class_id x_center y_center width height
                    parts[0] = '0'  # 将类别改为0
                    new_lines.append(' '.join(parts) + '\n')
            
            # 写回文件
            with open(label_file, 'w') as f:
                f.writelines(new_lines)
        
        print(f"  处理了 {len(label_files)} 个标注文件")

# 运行转换
convert_labels_to_single_class('/home/renyingcan/code/handdata_single')
print("标注转换完成！")