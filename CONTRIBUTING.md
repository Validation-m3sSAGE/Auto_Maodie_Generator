# 贡献指南

感谢你对本项目的关注！欢迎通过 Issue、Pull Request 等方式参与贡献，以下是参与指南。


## 提交 Issue

如果遇到问题、有功能建议或发现 Bug，请按以下格式提交 Issue：

1. **标题**：简明描述问题（如“Bug：替换纯色纹理时程序崩溃”或“建议：增加纹理缩放功能”）。
2. **内容**：
   - 若为 Bug：描述复现步骤、预期结果、实际结果，附上错误日志或截图（如有）。
   - 若为建议：说明功能用途、使用场景，帮助我们理解需求价值。
3. **环境信息**：Python 版本、OpenCV 版本、操作系统（如 Windows 10 / macOS 13）。


## 提交 Pull Request（PR）

如果你希望直接修改代码贡献功能，请遵循以下流程：

### 1. 分支规范
- 从 `main` 分支创建功能分支，命名格式：`feature/功能名称`（如 `feature/texture-scaling`）或 `fix/bug描述`（如 `fix/color-tolerance-crash`）。
- 确保分支与 `main` 同步最新代码，避免冲突。

### 2. PR 提交要求
- 标题清晰描述修改内容（如“新增纹理旋转功能”或“修复低光照场景下颜色识别错误”）。
- 内容说明修改动机、实现思路，关联相关 Issue（如 `Fixes #123`）。
- 确保所有代码通过本地测试，无语法错误或运行时异常。


## 开发环境设置

如需本地开发调试，可按以下步骤准备环境：

1. 克隆仓库并进入目录：
   ```bash
   git clone https://github.com/Validation-m3sSAGE/Auto_Maodie_Generator.git
   cd 视频纹理替换工具
   ```

2. 安装开发依赖：
   ```bash
   pip install numpy opencv-python
   ```


## 沟通方式

- 对于简单问题，可直接在 Issue 中留言讨论。
- 复杂功能建议或架构调整，建议先创建 Issue 发起讨论，达成共识后再开发。

期待你的贡献，让这个工具更完善！