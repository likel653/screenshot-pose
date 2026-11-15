"""Per-video owned inference resources. Never share engines across jobs."""
import queue,struct,subprocess,threading,time

class Engines:
    def __init__(self):self.resources={};self.closed=False;self.created={}
    def get(self,key,factory):
        if self.closed:raise RuntimeError('识别会话已关闭')
        if key not in self.resources:
            self.resources[key]=factory();name=str(key[0]);self.created[name]=self.created.get(name,0)+1
        return self.resources[key]
    def close(self):
        for value in reversed(list(self.resources.values())):
            close=getattr(value,'close',None)
            if close:close()
        self.resources.clear();self.closed=True

class NativeWorker:
    """Existing sam3d --worker framed path protocol, with bounded reads."""
    def __init__(self,command,env,log,timeout=180.):
        self.timeout=timeout;self.events=queue.Queue();self.log=log.open('wb');self.process=None;self.closed=False
        try:
            self.process=subprocess.Popen([str(x) for x in command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()
            if self.line()!=b'READY':raise RuntimeError('SAM 常驻运行器未返回 READY；请更新本地运行器')
        except Exception:self.close();raise
    def _read(self):
        try:
            while True:
                line=self.process.stdout.readline(4097)
                if not line:break
                if len(line)>4096:self.events.put(None);return
                self.events.put(line.strip())
        finally:self.events.put(None)
    def line(self):
        try:value=self.events.get(timeout=self.timeout)
        except queue.Empty:raise RuntimeError('SAM 常驻推理超时，请查看运行日志')
        if value is None:raise RuntimeError('SAM 常驻运行器意外退出，请查看运行日志')
        return value
    def infer(self,wire,result):
        try:
            for path in (wire,result):
                value=str(path).encode('utf8')
                if not 1<=len(value)<=4096 or b'\0' in value:raise ValueError('SAM 请求路径过长或含无效字符')
                self.process.stdin.write(struct.pack('<I',len(value))+value)
            self.process.stdin.flush();timing=None;deadline=time.monotonic()+self.timeout
            while True:
                if time.monotonic()>deadline:raise RuntimeError('SAM 常驻运行器请求超时')
                line=self.line()
                if line==b'DONE':
                    if timing is None or not result.is_file():raise RuntimeError('SAM 常驻结果不完整')
                    return timing
                if line.startswith(b'TIMING body_infer_ms '):timing=float(line.split()[-1])
                else:raise RuntimeError('SAM 常驻运行器返回了未知响应')
        except Exception:self.close();raise
    def close(self):
        if self.closed:return
        self.closed=True;p=self.process
        if p:
            if p.stdin:
                try:p.stdin.close()
                except OSError:pass
            try:p.wait(timeout=5.)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5.)
            if hasattr(self,'reader'):self.reader.join(timeout=2.)
            if p.stdout:p.stdout.close()
        self.log.close()
