import os
import json
import numpy as np
import cv2
from pathlib import Path

def safe_imwrite(path, img):
    """安全保存图像，支持中文路径"""
    path = str(path)  # 确保是字符串
    ext = os.path.splitext(path)[1]
    if not ext:
        path += ".png"  # 默认使用PNG格式
        ext = ".png"
    
    # 获取正确的编码器
    if ext.lower() in ['.jpg', '.jpeg']:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif ext.lower() == '.webp':
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), 95]
    else:  # PNG或其他格式
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 5]
    
    # 使用imencode保存
    success, buf = cv2.imencode(ext, img, encode_param)
    if success:
        with open(path, 'wb') as f:
            f.write(buf)
    else:
        raise IOError(f"无法保存图像: {path}")

def process_json_files(json_files, output_root):
    """处理多个JSON标注文件，生成掩码并裁剪二值图"""
    for json_path in json_files:
        print(f"处理文件: {json_path}")
        
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像尺寸
        width = data['imageWidth']
        height = data['imageHeight']
        
        # 处理每个标注形状
        for shape_index, shape in enumerate(data['shapes']):
            # 创建空白掩码
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # 获取多边形点并转换为整数坐标
            points = np.array(shape['points'], dtype=np.int32)
            
            # 在掩码上绘制多边形（填充为白色）
            cv2.fillPoly(mask, [points], 255)
            
            # 获取最小外接矩形
            x, y, w, h = cv2.boundingRect(points)
            if w == 0 or h == 0:  # 跳过无效矩形
                print(f"警告: 跳过无效矩形 (GUID: {shape['guid']})")
                continue
                
            # 裁剪二值图
            cropped_mask = mask[y:y+h, x:x+w]
            
            # 获取形态属性并创建对应目录
            attributes = shape.get('attributes', {})
            category = attributes.get('形态', 'unknown')
            if not category or not isinstance(category, str):
                category = "unknown"
                print(f"警告: 无效的分类属性 (索引: {shape_index})")

            category_dir = Path(output_root) / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成唯一文件名并保存
            base_name = Path(json_path).stem
            output_path = category_dir / f"{base_name}_{shape_index}_{shape['label']}.png"

            safe_imwrite(str(output_path), cropped_mask)
            print(f"保存裁剪二值图到: {output_path}")
        
        print(f"完成处理: {json_path}\n")

def find_json_files(directory):
    """查找目录中的所有JSON文件"""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


if __name__ == "__main__":
    # 配置路径
    json_dir = "G:/BGA项目/量产机/BGA-训练数据/模型4训练数据/4-1-1212/Seg"
    output_root = "E:/datasets/shape_classification"
    
    # 获取所有JSON文件
    json_files = find_json_files(json_dir)
    
    if not json_files:
        print(f"在目录 {json_dir} 中未找到JSON文件")
    else:
        print(f"找到 {len(json_files)} 个JSON文件")
        # 处理所有JSON文件
        process_json_files(json_files, output_root)
        print("所有文件处理完成！")