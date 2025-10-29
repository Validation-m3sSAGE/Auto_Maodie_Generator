import cv2
import numpy as np
import os
from tqdm import tqdm

def process_video(input_video_path, fill_image_path, output_video_path, 
                  white_threshold=180, min_circle_radius=5, max_circle_radius=100):
    """
    处理视频，用指定图片填充视频中的圆形白色区域（修复边界检查和尺寸匹配问题）
    """
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"无法打开输入视频: {input_video_path}")
        return
    
    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 定义输出视频编码器
    fourcc_options = ['mp4v', 'avc1', 'xvid', 'mjpg']
    out = None
    for fourcc_code in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            if out.isOpened():
                break
        except:
            continue
    
    if out is None or not out.isOpened():
        print(f"无法创建输出视频文件: {output_video_path}")
        cap.release()
        return
    
    # 读取填充图片
    fill_img = cv2.imread(fill_image_path)
    if fill_img is None:
        print(f"无法读取填充图片: {fill_image_path}")
        cap.release()
        out.release()
        return
    
    # 处理每一帧
    with tqdm(total=total_frames, desc="处理视频") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 双边滤波保留边缘
            gray_blur = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # 自适应阈值处理
            _, thresh = cv2.threshold(gray_blur.astype(np.uint8), white_threshold, 255, cv2.THRESH_BINARY)
            
            # 形态学闭运算填充孔洞
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 边缘检测
            edges = cv2.Canny(thresh, 50, 150)
            #frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            #'''
            # 霍夫圆检测
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=30,
                param1=80,
                param2=25,
                minRadius=min_circle_radius,
                maxRadius=max_circle_radius
            )
            
            # 处理检测到的圆
            if circles is not None:
                # 转换为整数并确保半径为正数
                circles = np.uint16(np.around(circles))[0, :]
                
                for circle in circles:
                    x, y, r = circle
                    
                    # 关键修复1：严格检查圆形是否完全在帧内（避免ROI尺寸异常）
                    # 确保x-r、x+r在宽度范围内，y-r、y+r在高度范围内
                    if (x - r < 0) or (x + r >= frame_width) or \
                       (y - r < 0) or (y + r >= frame_height):
                        continue  # 跳过超出边界的圆
                    
                    # 关键修复2：确保半径有效（避免0或负数半径）
                    if r <= 0 or 2*r > frame_width or 2*r > frame_height:
                        continue
                    
                    # 缩放填充图片到直径大小（2r x 2r）
                    scaled_fill = cv2.resize(fill_img, (2*r, 2*r), interpolation=cv2.INTER_AREA)
                    
                    # 创建圆形掩码（尺寸与缩放后的图片一致）
                    mask = np.zeros((2*r, 2*r, 3), dtype=np.uint8)
                    cv2.circle(mask, (r, r), r, (255, 255, 255), -1)
                    
                    # 提取ROI（此时ROI尺寸一定是2r x 2r，与mask匹配）
                    roi = frame[y-r:y+r, x-r:x+r]
                    
                    # 确保ROI尺寸正确（双重保险）
                    if roi.shape[0] != 2*r or roi.shape[1] != 2*r:
                        continue
                    
                    # 填充操作（此时尺寸匹配，不会报错）
                    filled_roi = cv2.bitwise_and(scaled_fill, mask)
                    roi_background = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
                    final_roi = cv2.add(filled_roi, roi_background)
                    frame[y-r:y+r, x-r:x+r] = final_roi
            #'''
            # 写入输出视频
            out.write(frame)
            pbar.update(1)
    
    print("处理完成")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='用指定图片填充视频中的圆形白色区域')
    parser.add_argument('--input_video', default="./input/cl1.mp4", help='输入视频路径')
    parser.add_argument('--fill_image', default="./texture/cc.png", help='用于填充的图片路径')
    parser.add_argument('--output_video', default="./output/cl1.mp4", help='输出视频路径')
    parser.add_argument('--threshold', type=int, default=180, help='白色区域的阈值，默认200')
    parser.add_argument('--min-radius', type=int, default=15, help='最小圆半径，默认5')
    parser.add_argument('--max-radius', type=int, default=35, help='最大圆半径，默认100')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_video):
        print(f"输入视频文件不存在: {args.input_video}")
    elif not os.path.exists(args.fill_image):
        print(f"填充图片文件不存在: {args.fill_image}")
    else:
        process_video(
            args.input_video, 
            args.fill_image, 
            args.output_video,
            args.threshold,
            args.min_radius,
            args.max_radius
        )