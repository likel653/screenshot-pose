# 运行与输出

`scripts/local_config.json` 为此机器的初始路径；运行参数可以覆盖模板和骨架。模型位置来自插件 `local_runtime.json` 的 runtime，找不到时回退项目 test_output/models。脚本加载源码目录中的插件，使用 `--factory-startup` 避免安装版污染测试。

输出目录必须不存在。包含：下载或复制的媒体、cache、job.json、run.log、blender.log、progress.json、project.blend、result.json、reopen_check.json，以及可选 preview.png。视频缓存和报告保留相同输出目录中的绝对路径；迁移文件夹后需重新载入缓存，不承诺所有外部动作缓存自动重定位。原材质贴图能打包时打包，不能打包的项目列出警告。

视频网页需要 yt-dlp 模块和通常需要 FFmpeg；不自动导出浏览器 cookies，不绕过登录/付费访问。直接媒体 URL 可用扩展名判定。模型本地齐全时默认不联网。链接和路径作为参数列表传递，不拼接 shell 命令。

仅复用已有缓存：`--kind cache --input <sequence.json>`。此模式复制缓存文件夹至新输出后再生成工程，报告注明复用缓存；适合调试骨架兼容性，不能用它冒充新模型识别测试。当前插件默认原地动画，保留相对姿态，不恢复场景中行走轨迹。
