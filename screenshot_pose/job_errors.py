"""Extract the actual worker failure instead of buffered model-load chatter."""
import re

def failure_message(text, code):
    lines = [re.sub(r'\x1b\[[0-9;]*m', '', line).strip() for line in text.splitlines() if line.strip()]
    # stdout can flush after stderr at process exit; the last line is unreliable.
    failures = [line for line in lines if re.match(r'^(?:视频识别失败|屏幕录制失败|识别失败|失败|[\w.]+(?:Error|Exception))[:：]', line)]
    if failures:
        return failures[-1]
    useful = [line for line in lines if not (line.startswith('load ') and ' backend' in line)]
    return useful[-1] if useful else f'进程退出 {code}，请打开日志查看详情'
