---
name: screenshot-pose-project
description: 使用本地 Screenshot Pose Blender 插件，把用户提供的图片、本地视频或可下载的视频链接转成角色姿势和关键帧，交付独立 .blend 工程、识别缓存与运行报告。用于自动摆姿和视频转角色动画。
---

# 图片或视频转角色工程

使用随附 `scripts/run_project.py` 完成实际识别、插件重定向和工程生成，不以写操作说明代替执行。它启动独立后台 Blender，不修改已打开窗口和源工程。

## 输入与默认选择

- 从用户消息/附件获取真实本地路径或 HTTP(S) 链接。若附件没有本地文件路径，先通过可用工具保存附件。
- 用户明确指定的角色、骨架和时间范围优先。未指定角色时使用 `scripts/local_config.json` 中配置的模板及骨架。其他角色若有多个骨架且无法确定，列出候选后向用户确认。
- 普通视频默认 SAM GPU、12 次/秒识别、24 FPS 工程；用户要求每帧时 `--sample-fps 0`，要求原视频帧率时先用 OpenCV 读取再传 `--fps`。未指定范围时处理整个视频，开始前报告时长和预计采样量；超过插件 10000 帧上限时协商或分段，不偷偷截取前几秒充当完整结果。
- 图片产生两帧相同的静态姿势，便于时间轴查看；视频保留采样时间，自动平滑与连续性复核沿用插件。
- 链接走 yt-dlp；直接图片链接可下载图片。不能获取时查看 `run.log` 说明具体原因，改用用户提供的本地视频，或在用户同意录制对应画面后用现有录屏流程。不要声称下载了实际没有拿到的媒体。参考 [references/workflow.md](references/workflow.md)。

## 执行

首次使用时，将 `scripts/local_config.example.json` 复制为 `scripts/local_config.json` 并填写本机路径。在源码工作区将 `screenshot_pose/local_runtime.example.json` 复制为 `screenshot_pose/local_runtime.json` 并填写识别环境。使用配置中的 Python 运行：

```powershell
& '<python>' '<skill>/scripts/run_project.py' --input '<图片或视频路径或链接>' --output '<新的绝对输出目录>'
```

可传 `--kind image|video|cache`、`--template`、`--rig`、`--start`、`--end`（秒）、`--sample-fps`、`--fps`、`--start-frame`、`--render`、`--keep-camera`。其他机器可用 `--config` 指定包含同名路径字段的 JSON。

长任务用进程会话运行，查看 `progress.json`、`cache/progress.json` 和日志，持续向用户报告已完成阶段。退出码非零或 `result.json` 不存在/`success` 非真都不是成功；诊断实际日志后修复，不绕过检测或把旧缓存说成新识别。默认不联网安装模型/依赖；缺失时明确指出具体路径并按用户授权准备环境。

## 交付与质量判断

- 成功后检查 `result.json` 的目标骨架、动作关键帧数、帧范围、缺帧/警告，以及 `reopen_check.json` 的工程重新加载验证。
- 若要求预览，使用 `--render`，查看真实渲染，至少检查脸部材质、相机内是否有角色。默认保留源场景，并创建覆盖首/中/末帧的全身预览相机；用户要求沿用镜头时传 `--keep-camera`。
- 交付 `project.blend` 与报告链接，简短说明角色、帧范围和明显识别问题。保留整个输出文件夹以便复核原图。骨骼有关键帧不等于动作识别正确；不要宣称从单目视频恢复了真实三维或绝对人物位移。
- 不自动加锁脚/扶桌区间：接触须有用户指示或明确证据及目视复核，避免误锁跳跃。手工修正可沿用插件 v0.25/0.26 的区间预览流程。
- 用户要求“当前未保存场景”时，不能用磁盘模板冒充。可靠连接当前 Blender 读取/另存副本，或请用户保存到给定路径，再运行脚本。
