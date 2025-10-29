import cv2
import numpy as np
import argparse

# -------------------------- 1. 16色生成（4×4） --------------------------
def create_color_texture_map():
    color_list = []
    # 生成16种颜色（仅执行一次）
    for norm_x in range(4):
        for norm_y in range(4):
            r = 80 + norm_x * 30
            g = 80 + norm_y * 30
            b = 360 - r - g
            color_list.append((r, g, b))  # 保持(R,G,B)顺序
    
    np.random.seed(432)
    texture_indices = np.random.randint(0, 13, size=len(color_list)).tolist()  # 16个索引
    return color_list, texture_indices

# -------------------------- 2. 纹理加载与初始化预处理（核心优化） --------------------------
def load_textures_and_preprocess(target_width, target_height, color_list, texture_indices, texture_dir="./texture"):
    """初始化时一次性完成所有重复操作：加载纹理→转换数组→预处理映射关系"""
    # 1. 加载纹理（仅执行一次）
    texture_paths = [f"{texture_dir}/{i}.png" for i in range(1, 14)]
    textures = []
    for path in texture_paths:
        tex = cv2.imread(path)
        if tex is None:
            raise FileNotFoundError(f"❌ 无法加载纹理图：{path}")
        if tex.shape[0] != target_height or tex.shape[1] != target_width:
            tex = cv2.resize(tex, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        textures.append(tex)
    print(f"✅ 成功加载{len(textures)}张纹理图")
    
    # 2. 预处理：颜色列表→数组（仅执行一次）
    color_array = np.array(color_list, dtype=np.uint8)[:, np.newaxis, np.newaxis, :]  # (16, 1, 1, 3)
    
    # 3. 预处理：纹理列表→数组+索引映射（仅执行一次）
    texture_array = np.array(textures, dtype=np.uint8)  # (13, H, W, 3)
    color_textures = texture_array[texture_indices]  # (16, H, W, 3)（预计算16色对应的纹理）
    
    return color_array, color_textures  # 返回预处理结果，供后续帧直接使用

# -------------------------- 3. 纹理转换（仅含逐帧变化的操作） --------------------------
def color_to_texture_frame(color_frame, color_array, color_textures):
    """仅处理每帧变化的部分：生成掩码→计算纹理区域→叠加"""
    height, width = color_frame.shape[:2]
    
    # 生成掩码（仅随帧变化）
    all_masks = np.all(color_frame[np.newaxis, ...] == color_array, axis=3)  # (16, H, W)
    all_masks_uint8 = all_masks.astype(np.uint8) * 255
    all_masks_uint8 = all_masks_uint8[..., np.newaxis]  # (16, H, W, 1)
    
    # 计算纹理区域并叠加（复用初始化的color_textures）
    all_texture_regions = all_masks_uint8 * (255 - color_textures)
    texture_frame = np.sum(all_texture_regions, axis=0, dtype=np.uint16)
    return np.clip(texture_frame, 0, 255).astype(np.uint8)

# -------------------------- 4. 辅助函数 --------------------------
def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return (0, 0)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def get_position_based_color(center, frame_width, frame_height):
    total = 360
    norm_x = int(abs((center[0] * 2 / frame_width) - 1) * 4)
    norm_y = int(abs((center[1] * 2 / frame_height) - 1) * 4)
    norm_x = np.clip(norm_x, 0, 3)
    norm_y = np.clip(norm_y, 0, 3)
    
    r = 80 + norm_x * 30
    g = 80 + norm_y * 30
    b = 360 - r - g
    return (int(r), int(g), int(b))

# -------------------------- 5. 主函数（初始化与逐帧处理分离） --------------------------
def process_video(input_path, output_path, min_area=200):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{input_path}")
        return False

    # 视频基础信息（仅获取一次）
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 视频信息：宽={frame_width}, 高={frame_height}, 帧率={fps:.2f}, 总帧={total_frames}")

    # -------------------------- 初始化阶段：所有重复操作仅执行一次 --------------------------
        # 1. 生成颜色列表和纹理索引（固定不变）
    color_list, texture_indices = create_color_texture_map()
    # 2. 加载纹理并预处理（颜色数组、纹理映射等固定操作）
    color_array, color_textures = load_textures_and_preprocess(
        target_width=frame_width,
        target_height=frame_height,
        color_list=color_list,
        texture_indices=texture_indices
    )

    # 视频写入器（仅初始化一次）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"❌ 无法创建输出视频：{output_path}")
        cap.release()
        return False

    # 边缘检测工具（仅初始化一次）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kernel = np.ones((3, 3), np.uint8)
    frame_count = 0

    # -------------------------- 逐帧处理：仅执行变化的操作 --------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100
            print(f"🔄 进度：{frame_count}/{total_frames} 帧 ({progress:.1f}%)")

        # 边缘检测（逐帧变化）
        #'''
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        img_f = frame.astype(np.float32)
        sobel_x = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        sobel_x_sq = cv2.multiply(sobel_x, sobel_x)
        sobel_y_sq = cv2.multiply(sobel_y, sobel_y)
        sobel_mag = cv2.sqrt(sobel_x_sq + sobel_y_sq)
        gray_mag = np.max(sobel_mag, axis=2)
        gray_mag = cv2.GaussianBlur(gray_mag, (5, 5), 0)
        gray_mag = np.clip(gray_mag, 0, 255)
        _, edges = cv2.threshold(gray_mag.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        #output_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        #out.write(output_frame)
        '''
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_channel = hsv_frame[:, :, 1]
        v_channel = hsv_frame[:, :, 2]

        s_enhanced = clahe.apply(s_channel)
        s_blurred = cv2.GaussianBlur(s_enhanced, (5, 5), 0)
        s_edges = cv2.Canny(s_blurred, threshold1=3, threshold2=80)

        v_enhanced = clahe.apply(v_channel)
        v_blurred = cv2.GaussianBlur(v_enhanced, (5, 5), 0)
        v_edges = cv2.Canny(v_blurred, threshold1=3, threshold2=80)

        merged_edges = cv2.bitwise_or(s_edges, v_edges)
        merged_edges = cv2.dilate(merged_edges, kernel, iterations=1)
        merged_edges = cv2.erode(merged_edges, kernel, iterations=1)

        edges = merged_edges
        #'''
        # 染色（逐帧变化）
        color_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            contour = contours[i]
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            contour_hierarchy = hierarchy[0][i] if hierarchy is not None else [0, 0, 0, -1]
            is_outer = (contour_hierarchy[3] == -1)
            is_inner = (contour_hierarchy[3] != -1) and (contour_hierarchy[2] == -1)
            if not (is_outer or is_inner):
                continue
            center = get_contour_center(contour)
            block_color = get_position_based_color(center, frame_width, frame_height)
            cv2.drawContours(color_canvas, [contour], -1, block_color, thickness=-1)
            cv2.drawContours(color_canvas, [contour], -1, block_color, thickness=2)

        output_frame = color_canvas

        # 纹理转换
        final_texture_frame = color_to_texture_frame(output_frame, color_array, color_textures)
        out.write(final_texture_frame)
        #'''
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"🎉 处理完成！输出文件：{output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="初始化优化的纹理视频生成器")
    parser.add_argument('--input', type=str, default="./input/hs3.mp4", help='输入视频路径')
    parser.add_argument('--output', type=str, default="./output/hs3.mp4", help='输出纹理视频路径')
    parser.add_argument('--min-area', type=int, default=150, help="最小染色区域面积")
    args = parser.parse_args()
    process_video(args.input, args.output, args.min_area)
