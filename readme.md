# 耄耋自动生成器

一个基于 OpenCV-Python 的视频纹理替换工具，利用边缘检测划分视频中主体区域，并批量替换为耄耋皮肤。不是一键包！不是一键包！不是一键包！


## 示例效果

[哈气星魂](https://www.bilibili.com/video/BV1BZsiz2ETF)


## 功能介绍

- 支持读取本地MP4视频文件
- 输出处理后的视频文件，保留原视频编码格式


## 环境依赖

- Python 3.7+
- 核心依赖库：
  - `numpy`：用于数值计算
  - `opencv-python`（即 `cv2`）：用于视频处理和图像处理
> 注意：如果你是不知道什么是python，什么是conda，什么是命令行的哈基路人，这个项目可能不适合你。

## 快速开始

### 1. 安装依赖

打开终端/命令行，执行以下命令安装所需库（因为只有两个包所以用不用conda虚拟环境都无所谓）：

```bash
pip install numpy opencv-python
```

### 2. 克隆仓库

```bash
git clone https://github.com/Validation-m3sSAGE/Auto_Maodie_Generator.git
cd Auto_Maodie_Generator
```
### 3. 使用

#### 功能1：耄耋皮肤生成

```bash
python generator.py --input input.mp4 --output output.mp4 --min-area 150
```

#### 功能2：白色圆形区域替换

```bash
python circle.py --input input.mp4 --output output.mp4 --threshold 180 --min-radius 15 --max-radius 35
```

## 参数说明

| 参数名           | 类型          | 说明                                  |
|------------------|---------------|---------------------------------------|
| `input_video`    | str           | 输入视频文件路径（必填）              |
| `output_video`   | str           | 输出视频文件路径（必填）              |

其他参数可以在help中查看

```bash
python generator.py --help
python circle.py --help
```

## 注意事项

1. 视频处理速度取决于视频分辨率和CPU设备性能（约1秒两帧的处理速度），建议先测试短片段或低帧率片段；
2. 输出视频为黑色背景，预览视频中的效果是通过后期视频剪辑得到；
3. 因为使用的不是深度学习工具，只有视频编解码用显卡，所以你显卡再好也不会变得更快，别朝我哈气。


## 许可证

本项目基于 MIT 许可证开源，详情见 [LICENSE](LICENSE) 文件。


## 贡献

欢迎提交 Issue 或 Pull Request 改进工具功能，贡献前请先阅读 [贡献指南](CONTRIBUTING.md)。