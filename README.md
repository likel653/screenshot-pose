# Screenshot Pose

Screenshot Pose 是一个 Blender 插件。它在本机识别截图或视频中的人体姿势，将动作重定向到指定角色骨架，并生成可编辑的关键帧。

- 用参考图或框选截图识别姿势，可在 Blender 里校正关节点并存入本地姿势库。
- 用本地视频或录屏生成角色动画，支持自适应采样、逐帧贴合原图和脚部支撑。
- MediaPipe 可用于基础识别。SAM 3D Body、RTMW 和 HaMeR 提供可选的身体、腿脚和手部精修。

## 安装

在项目根目录运行 `python build_package.py` 生成插件 ZIP。然后在 Blender 的扩展或插件偏好设置中选择「从磁盘安装」，启用 Screenshot Pose。打开 3D 视图侧栏，选择目标骨架，运行「准备本地识别环境」。

GPU 模型需要单独准备。使用 SAM 时，在插件偏好设置中填写原生运行器所在目录，在识别环境中准备 GGUF 权重。可选模型及上游地址见 [第三方依赖与许可](screenshot_pose/THIRD_PARTY.txt)。如需在开发机保存运行环境路径，可参考 [本地配置示例](screenshot_pose/local_runtime.example.json)。

[中文使用说明](README_中文.md) · [命令行工程生成集成](integrations/screenshot-pose-project/SKILL.md)

## 第三方项目

| 用途 | 上游项目 |
| --- | --- |
| 基础身体、脸部和手部关键点 | [Google MediaPipe](https://github.com/google-ai-edge/mediapipe) |
| 全身和手部关键点、人体检测及 ONNX 推理 | [OpenMMLab MMPose（RTMW、RTMW3D、RTMPose Hand）](https://github.com/open-mmlab/mmpose)、[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)、[RTMLib](https://github.com/Tau-J/rtmlib)、[ONNX Runtime](https://github.com/microsoft/onnxruntime) |
| 可选三维身体重建 | [Meta SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)、[LocalAI 的 GGUF 转换](https://huggingface.co/LocalAI-io/sam-3d-body-dinov3-GGUF)及 [sam3d.cpp](https://github.com/localai-org/sam3d.cpp) |
| 可选三维手部重建 | [HaMeR](https://github.com/geopavlakos/hamer)、[WiLoR](https://github.com/rolpotamias/WiLoR)、[Hand Texture Module](https://github.com/gkarv/Hand-Texture-Module)、[MANO](https://mano.is.tue.mpg.de/) |
| 可选头部朝向、相机标定和视频脚部接触修正 | [6DRepNet360](https://github.com/thohemp/6DRepNet360)、[AnyCalib](https://github.com/javrtg/AnyCalib)、[RGB MoCap](https://github.com/toyxyz/toyxyz_RGBMocap) |

Screenshot Pose 的原创代码采用 [MIT 许可证](LICENSE)。安装包随附 MediaPipe 面部模板和 MHR 骨架数据的衍生文件及其许可。模型权重、MANO 数据和可选运行器由各上游单独提供，使用时适用各自条款。完整来源、用途与许可见 [第三方声明](screenshot_pose/THIRD_PARTY.txt)。

## 测试

在插件的识别环境中运行 `python -m unittest discover -s tests -p "test_*.py"`。使用本地模型或回归素材的用例在缺少相应文件时跳过。
