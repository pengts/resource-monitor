#!/usr/bin/env python3
"""轻量终端资源监控: CPU / 内存 / 磁盘 / 网络 / GPU 与进程 TopN.

CPU 与内存按本进程所在 cgroup 的实际限额统计 (容器内即容器配额), 无限额时按整机统计.
"""
import argparse
import heapq
import os
import pwd
import select
import socket
import sys
import termios
import time
import tty
from collections import deque
from types import SimpleNamespace

import psutil
import pynvml
from rich import box
from rich.cells import cell_len
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PAGE_SIZE = os.sysconf('SC_PAGE_SIZE')
CLK_TCK = os.sysconf('SC_CLK_TCK')
SPARK = '▁▂▃▄▅▆▇█'
HISTORY = 240
DISK_EVERY = 10.0  # 磁盘容量变化慢, 每 10 秒刷新一次
NARROW = 110       # 终端宽度低于此值时面板改为单列
MAX_NICS = 6       # 网卡过多 (如大量 veth) 时只显示当前速率最高的若干个


# ---------------------------------------------------------------- cgroup

def _read(path):
    with open(path) as f:
        return f.read().strip()


def _read_int(path):
    return int(_read(path))


def _read_kv(path):
    with open(path) as f:
        return {k: int(v) for k, v in (line.split() for line in f)}


def _cgroup_mounts():
    """解析本进程各 controller 的 cgroup 目录: {key: (挂载点, 本进程目录, 挂载根)}, v2 的 key 为 'v2'."""
    paths = {}
    with open('/proc/self/cgroup') as f:
        for line in f:
            hid, ctrls, path = line.rstrip('\n').split(':', 2)
            for k in (['v2'] if hid == '0' else ctrls.split(',')):
                paths[k] = path
    out = {}
    with open('/proc/self/mountinfo') as f:
        for line in f:
            pre, post = line.split(' - ', 1)
            fstype, _src, sopts = post.split()
            if fstype == 'cgroup2':
                keys = ['v2']
            elif fstype == 'cgroup':
                keys = [o for o in sopts.split(',') if o in ('cpu', 'cpuacct', 'memory')]
            else:
                continue
            root, mnt = pre.split()[3:5]
            for k in keys:
                if k in out or k not in paths:
                    continue
                rel = os.path.relpath(paths[k], root)
                if rel == '..' or rel.startswith('../'):
                    raise RuntimeError(f'进程 cgroup 路径 {paths[k]} 不在挂载根 {root} ({mnt}) 之下, 无法定位限额')
                d = os.path.normpath(os.path.join(mnt, rel))
                if not os.path.isdir(d):
                    raise RuntimeError(f'cgroup 目录不存在: {d}')
                out[k] = (mnt, d, root)
    return out


def _ancestors(mnt, d):
    """从 d 向上到挂载点 mnt (含) 的所有 cgroup 目录."""
    while True:
        yield d
        if d == mnt:
            return
        d = os.path.dirname(d)


def _v2_cpu_limit(d):
    p = os.path.join(d, 'cpu.max')
    if not os.path.exists(p):  # 根 cgroup 或未启用 cpu 控制器: 该层无限额
        return None
    quota, period = _read(p).split()
    return None if quota == 'max' else int(quota) / int(period)


def _v2_max(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    v = _read(p)
    return None if v == 'max' else int(v)


def _v1_cpu_limit(d):
    quota = _read_int(os.path.join(d, 'cpu.cfs_quota_us'))
    return None if quota < 0 else quota / _read_int(os.path.join(d, 'cpu.cfs_period_us'))


def _pick(mount, read_limit, host_value, in_container):
    """在本进程 cgroup 及可见祖先中取最严格且小于整机容量的限额.

    返回 (用量读取目录, 限额, 是否受限). 无限额时: 处于容器自身 cgroup 则读容器用量, 否则目录为 None (按整机).
    """
    mnt, d, _ = mount
    best = None
    for a in _ancestors(mnt, d):
        v = read_limit(a)
        if v is not None and v < host_value and (best is None or v < best[1]):
            best = (a, v)
    if best:
        return best[0], best[1], True
    return (mnt if in_container else None), host_value, False


def _cg_path(mount, d):
    mnt, _, root = mount
    return os.path.normpath(os.path.join(root, os.path.relpath(d, mnt)))


def detect_cgroup():
    cpu_n = len(os.sched_getaffinity(0))
    mem_total = psutil.virtual_memory().total
    cg = SimpleNamespace(ver=None, cpu_dir=None, cpu_limit=float(psutil.cpu_count()), mem_dir=None,
                         mem_limit=mem_total, swap_limit=None, swap_note=None)
    mounts = _cgroup_mounts()
    v2 = mounts.get('v2')
    if v2 and {'cpu', 'memory'} <= set(_read(os.path.join(v2[0], 'cgroup.controllers')).split()):
        cg.ver = 'v2'
        cpu_m = mem_m = v2
        # 真正的根 cgroup 没有 memory.current; 挂载点存在该文件说明挂载的是容器自身的 cgroup
        cpu_ct = mem_ct = os.path.exists(os.path.join(v2[0], 'memory.current'))
        cpu_lim, mem_lim = _v2_cpu_limit, lambda a: _v2_max(a, 'memory.max')
    elif {'cpu', 'cpuacct', 'memory'} <= mounts.keys():
        if mounts['cpu'] != mounts['cpuacct']:
            raise RuntimeError('cgroup v1 下 cpu 与 cpuacct 未共同挂载, 暂不支持')
        cg.ver = 'v1'
        cpu_m, mem_m = mounts['cpu'], mounts['memory']
        cpu_ct, mem_ct = cpu_m[2] != '/', mem_m[2] != '/'
        cpu_lim, mem_lim = _v1_cpu_limit, lambda a: _read_int(os.path.join(a, 'memory.limit_in_bytes'))
    else:
        cg.scope = (f'CPU: 整机 {cg.cpu_limit:g} 核   内存: 整机 {fmt_bytes(mem_total)}'
                    '   (未找到启用 cpu/memory 控制器的 cgroup)')
        return cg

    d, lim, limited = _pick(cpu_m, cpu_lim, cpu_n, cpu_ct)
    cpu_path = None if d is None else _cg_path(cpu_m, d)
    if d is None:
        cpu_desc = f'整机 {cg.cpu_limit:g} 核'
    else:
        cg.cpu_dir, cg.cpu_limit = d, float(lim)
        cpu_desc = f'限额 {lim:.2f} 核' if limited else f'无限额, 按可用 {lim} 核'

    d, lim, limited = _pick(mem_m, mem_lim, mem_total, mem_ct)
    mem_path = None if d is None else _cg_path(mem_m, d)
    if d is None:
        mem_desc = f'整机 {fmt_bytes(mem_total)}'
    else:
        cg.mem_dir, cg.mem_limit = d, lim
        mem_desc = f'限额 {fmt_bytes(lim)}' if limited else f'无限额, 按整机 {fmt_bytes(lim)}'
        if cg.ver == 'v1':
            cg.swap_note = 'cgroup v1 下不统计'
        elif not os.path.exists(os.path.join(d, 'memory.swap.current')):
            cg.swap_note = 'cgroup 未开启 swap 统计'
        else:
            limits = [v for a in _ancestors(mem_m[0], d) if (v := _v2_max(a, 'memory.swap.max')) is not None]
            cg.swap_limit = min([psutil.swap_memory().total] + limits)

    if cpu_path == mem_path:
        cg.scope = (f'cgroup {cg.ver} {cpu_path}   ' if cpu_path else '') + f'CPU: {cpu_desc}   内存: {mem_desc}'
    else:
        cg.scope = (f'CPU: {cpu_desc}' + (f' (cgroup {cg.ver} {cpu_path})' if cpu_path else '')
                    + f'   内存: {mem_desc}' + (f' (cgroup {cg.ver} {mem_path})' if mem_path else ''))
    return cg


# ---------------------------------------------------------------- GPU

def _nvml_opt(fn, *a):
    """部分 GPU 不支持温度/功耗等查询, 此时返回 None 并显示为 '-'."""
    try:
        return fn(*a)
    except pynvml.NVMLError_NotSupported:
        return None


class Gpu:
    def __init__(self):
        self.handles = []
        try:
            pynvml.nvmlInit()
        except (pynvml.NVMLError_LibraryNotFound, pynvml.NVMLError_DriverNotLoaded) as e:
            self.status = f'未检测到可用 NVIDIA GPU ({e})'
            return
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self.names = [pynvml.nvmlDeviceGetName(h) for h in self.handles]
        self.status = None if self.handles else 'NVML 可用, 但未发现 GPU 设备'

    def sample(self):
        devs, procs = [], []
        for i, h in enumerate(self.handles):
            util = _nvml_opt(pynvml.nvmlDeviceGetUtilizationRates, h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            devs.append(SimpleNamespace(
                idx=i, name=self.names[i], util=None if util is None else util.gpu,
                mem_used=mem.used, mem_total=mem.total,
                temp=_nvml_opt(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU),
                power=_nvml_opt(pynvml.nvmlDeviceGetPowerUsage, h),
                power_limit=_nvml_opt(pynvml.nvmlDeviceGetEnforcedPowerLimit, h)))
            seen = {}
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h) + pynvml.nvmlDeviceGetGraphicsRunningProcesses(h):
                seen.setdefault(p.pid, p.usedGpuMemory)  # usedGpuMemory 为 None 表示驱动无法提供
            procs += [(i, pid, m) for pid, m in seen.items()]
        return devs, procs


# ---------------------------------------------------------------- 采样

def scan_procs():
    """直接读 /proc/[pid]/stat: {pid: (starttime, utime+stime, rss, comm)}. 比 psutil 逐进程查询快约 4 倍."""
    out = {}
    for name in os.listdir('/proc'):
        if not name.isdigit():
            continue
        try:
            with open(f'/proc/{name}/stat', 'rb') as f:
                s = f.read()
        except (FileNotFoundError, ProcessLookupError):
            continue  # 进程在列目录之后已退出
        rp = s.rindex(b')')
        fl = s[rp + 2:].split()
        out[int(name)] = (int(fl[19]), int(fl[11]) + int(fl[12]), int(fl[21]) * PAGE_SIZE,
                          s[s.index(b'(') + 1:rp])
    return out


class Sampler:
    def __init__(self, cg, gpu, top_n):
        self.cg, self.gpu, self.top_n = cg, gpu, top_n
        self.cpus = sorted(os.sched_getaffinity(0))
        self.boot = psutil.boot_time()
        self.hist = {k: deque(maxlen=HISTORY) for k in ('cpu', 'mem', 'rx', 'tx')}
        self.users = {}
        self.disks, self.disks_t = [], -DISK_EVERY
        psutil.cpu_percent(percpu=True)  # 建立基线
        self.prev = self._counters()

    def _counters(self):
        cg_cpu = None
        if self.cg.cpu_dir and self.cg.ver == 'v2':
            cg_cpu = _read_kv(os.path.join(self.cg.cpu_dir, 'cpu.stat'))['usage_usec'] / 1e6
        elif self.cg.cpu_dir:
            cg_cpu = _read_int(os.path.join(self.cg.cpu_dir, 'cpuacct.usage')) / 1e9
        return SimpleNamespace(t=time.monotonic(), cg_cpu=cg_cpu, procs=scan_procs(),
                               net=psutil.net_io_counters(pernic=True),
                               io=psutil.disk_io_counters(perdisk=True), self_cpu=sum(os.times()[:2]))

    def _memory(self):
        cg = self.cg
        if cg.mem_dir is None:
            vm = psutil.virtual_memory()
            used, cache = vm.total - vm.available, vm.buffers + vm.cached
        elif cg.ver == 'v2':
            st = _read_kv(os.path.join(cg.mem_dir, 'memory.stat'))
            used = _read_int(os.path.join(cg.mem_dir, 'memory.current')) - st['inactive_file']
            cache = st['file']
        else:
            st = _read_kv(os.path.join(cg.mem_dir, 'memory.stat'))
            used = _read_int(os.path.join(cg.mem_dir, 'memory.usage_in_bytes')) - st['total_inactive_file']
            cache = st['total_cache']
        if cg.swap_note:
            swap = None
        elif cg.mem_dir is None:
            sw = psutil.swap_memory()
            swap = (sw.used, sw.total)
        else:
            swap = (_read_int(os.path.join(cg.mem_dir, 'memory.swap.current')), cg.swap_limit)
        return used, cache, swap

    def _refresh_disks(self):
        """物理文件系统按设备去重 (保留最短挂载点), 并始终包含根目录 (容器内为 overlay)."""
        parts = {}
        for p in psutil.disk_partitions(all=False):
            if os.path.isdir(p.mountpoint) and (p.device not in parts or len(p.mountpoint) < len(parts[p.device])):
                parts[p.device] = p.mountpoint
        if '/' not in parts.values():
            root = next(p for p in psutil.disk_partitions(all=True) if p.mountpoint == '/')
            parts[root.device] = '/'
        self.disks = sorted(((mnt, dev, psutil.disk_usage(mnt)) for dev, mnt in parts.items()))

    def _detail(self, pid, comm):
        try:
            uid = os.stat(f'/proc/{pid}').st_uid
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = ' '.join(f.read().decode(errors='replace').replace('\0', ' ').split())  # 合并换行等空白, 保证单行
        except (FileNotFoundError, ProcessLookupError):
            return '-', '(已退出)'
        if uid not in self.users:
            try:
                self.users[uid] = pwd.getpwuid(uid).pw_name
            except KeyError:
                self.users[uid] = str(uid)  # 与 ps 一致, 无对应用户名时显示 uid
        return self.users[uid], cmd or f'[{comm.decode(errors="replace")}]'

    def sample(self):
        cg, prev = self.cg, self.prev
        cur = self.prev = self._counters()
        dt = cur.t - prev.t
        s = SimpleNamespace(dt=dt)

        percpu = psutil.cpu_percent(percpu=True)
        s.cores = [(i, percpu[i]) for i in self.cpus]
        s.cpu_used = sum(percpu) / 100 if cg.cpu_dir is None else (cur.cg_cpu - prev.cg_cpu) / dt
        s.cpu_pct = s.cpu_used / cg.cpu_limit * 100
        s.load = os.getloadavg()
        s.uptime = time.time() - self.boot

        s.mem_used, s.mem_cache, s.swap = self._memory()
        s.mem_pct = s.mem_used / cg.mem_limit * 100
        self.hist['cpu'].append(s.cpu_pct)
        self.hist['mem'].append(s.mem_pct)

        if cur.t - self.disks_t >= DISK_EVERY:
            self._refresh_disks()
            self.disks_t = cur.t
        s.disks = []
        for mnt, dev, du in self.disks:
            name = os.path.basename(dev)
            a, b = cur.io.get(name), prev.io.get(name)
            rw = None if a is None or b is None else ((a.read_bytes - b.read_bytes) / dt,
                                                       (a.write_bytes - b.write_bytes) / dt)
            s.disks.append((mnt, dev, du, rw))

        stats = psutil.net_if_stats()
        nics, rx, tx = [], 0.0, 0.0
        for nic, a in cur.net.items():
            st = stats.get(nic)  # None: 网卡在两次读取之间被移除 (如容器 veth)
            if nic == 'lo' or st is None or not st.isup:
                continue
            b = prev.net.get(nic)
            rate = None if b is None else ((a.bytes_recv - b.bytes_recv) / dt, (a.bytes_sent - b.bytes_sent) / dt)
            if rate:
                rx, tx = rx + rate[0], tx + rate[1]
            nics.append((nic, rate, a.bytes_recv, a.bytes_sent))
        self.hist['rx'].append(rx)
        self.hist['tx'].append(tx)
        shown = heapq.nlargest(MAX_NICS, nics, key=lambda r: -1 if r[1] is None else r[1][0] + r[1][1])
        s.nics, s.nics_hidden = sorted(shown), len(nics) - len(shown)
        s.hist = self.hist

        rows = []
        for pid, (start, ticks, rss, comm) in cur.procs.items():
            p = prev.procs.get(pid)
            d = ticks - p[1] if p and p[0] == start else ticks  # 不在上轮快照中的进程在本周期内启动
            rows.append((pid, d / CLK_TCK / dt * 100, rss, comm))
        n = self.top_n
        top_cpu = heapq.nlargest(n, rows, key=lambda r: r[1])
        top_mem = heapq.nlargest(n, rows, key=lambda r: r[2])
        details = {r[0]: self._detail(r[0], r[3]) for r in top_cpu + top_mem}
        s.top_cpu = [r + details[r[0]] for r in top_cpu]
        s.top_mem = [r + details[r[0]] for r in top_mem]

        s.gpus, s.gpu_procs = [], []
        if self.gpu.handles:
            s.gpus, gp = self.gpu.sample()
            gp = heapq.nlargest(n, gp, key=lambda r: -1 if r[2] is None else r[2])
            for gi, pid, used in gp:
                if pid in cur.procs:
                    user, cmd = self._detail(pid, cur.procs[pid][3])
                else:
                    user, cmd = '-', '(不在当前 PID 命名空间)'
                s.gpu_procs.append((gi, pid, used, user, cmd))

        s.self_cpu = (cur.self_cpu - prev.self_cpu) / dt * 100
        s.self_rss = cur.procs[os.getpid()][2]
        return s


# ---------------------------------------------------------------- 渲染

def fmt_bytes(n):
    for unit in 'BKMGT':
        if n < 1024:
            return f'{n:.0f}{unit}' if unit == 'B' else f'{n:.1f}{unit}'
        n /= 1024
    return f'{n:.1f}P'


def fmt_rate(n):
    return fmt_bytes(n) + '/s'


def fmt_dur(sec):
    d, r = divmod(int(sec), 86400)
    h, r = divmod(r, 3600)
    return f'{d}天 {h:02d}:{r // 60:02d}' if d else f'{h:02d}:{r // 60:02d}'


def level(pct):
    return 'green' if pct < 60 else 'yellow' if pct < 85 else 'red'


def bar(pct, width):
    filled = round(min(max(pct, 0.0), 100.0) / 100 * width)  # 仅限制填充长度, 数值照实显示
    return Text.assemble(('█' * filled, level(pct)), ('░' * (width - filled), 'grey30'))


def spark(values, width, top=100.0, style=None):
    vals = list(values)[-width:]
    t = Text(' ' * (width - len(vals)))
    for v in vals:
        i = min(int(v / top * (len(SPARK) - 1) + 0.5), len(SPARK) - 1) if top > 0 else 0
        t.append(SPARK[i], style or level(v / top * 100))
    return t


def line(*parts):
    return Text.assemble(*parts, no_wrap=True, overflow='ellipsis')


def panel(body, title):
    return Panel(body, title=f'[bold cyan]{title}', title_align='left', border_style='grey42',
                 box=box.ROUNDED, padding=(0, 1))


def table(*cols):
    t = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, pad_edge=False,
              header_style='bold', border_style='grey42', padding=(0, 1))
    for name, kw in cols:
        t.add_column(name, no_wrap=True, **kw)
    return t


R = {'justify': 'right'}
CMD = {'ratio': 1, 'overflow': 'ellipsis'}


def cpu_body(s, cg, inner):
    used = f'  {s.cpu_used:.2f}/{cg.cpu_limit:g} 核'
    lines = [
        line('总占用 ', bar(s.cpu_pct, max(inner - 14 - cell_len(used), 5)), f' {s.cpu_pct:5.1f}%', (used, 'dim')),
        line('趋势   ', spark(s.hist['cpu'], inner - 7)),
        line('负载   ', f'{s.load[0]:.2f}  {s.load[1]:.2f}  {s.load[2]:.2f}',
             ('  (1/5/15 分钟' + (', 整机)' if cg.cpu_dir else ')'), 'dim')),
    ]
    cols = max(1, inner // 16)
    if len(s.cores) > cols * 8:  # 核数过多时用单字符热度条, 避免面板过高
        per = inner - 7
        for k in range(0, len(s.cores), per):
            chunk = s.cores[k:k + per]
            lines.append(line(f'{chunk[0][0]:>4}-  ', spark([p for _, p in chunk], len(chunk))))
        return Group(*lines), len(lines)
    bw = inner // cols - 10
    grid = Table.grid(padding=(0, 1))
    for _ in range(cols):
        grid.add_column(no_wrap=True)
    cells = [line(f'{i:>3} ', bar(p, bw), f'{p:4.0f}%') for i, p in s.cores]
    for k in range(0, len(cells), cols):
        grid.add_row(*cells[k:k + cols])
    return Group(*lines, grid), len(lines) + (len(cells) + cols - 1) // cols


def mem_body(s, cg, inner):
    total = f'  {fmt_bytes(s.mem_used)}/{fmt_bytes(cg.mem_limit)}'
    lines = [line('内存   ', bar(s.mem_pct, max(inner - 14 - cell_len(total), 5)), f' {s.mem_pct:5.1f}%', (total, 'dim')),
             line('缓存   ', fmt_bytes(s.mem_cache), ('  (可回收的文件缓存不计入已用)', 'dim'))]
    if s.swap is None:
        lines.append(line('Swap   ', (cg.swap_note, 'dim')))
    elif s.swap[1] == 0:
        lines.append(line('Swap   ', ('未配置', 'dim')))
    else:
        pct = s.swap[0] / s.swap[1] * 100
        txt = f'  {fmt_bytes(s.swap[0])}/{fmt_bytes(s.swap[1])}'
        lines.append(line('Swap   ', bar(pct, max(inner - 14 - cell_len(txt), 5)), f' {pct:5.1f}%', (txt, 'dim')))
    lines.append(line('趋势   ', spark(s.hist['mem'], inner - 7)))
    return Group(*lines), len(lines)


RATE = {'justify': 'right', 'min_width': 9}  # 速率列固定最小宽度, 避免数值变化导致表格抖动


def disk_body(s, inner):
    t = table(('挂载点', {}), ('用量', {}), ('已用/总量', R), ('读', RATE), ('写', RATE))
    used = [f'{fmt_bytes(du.used)}/{fmt_bytes(du.total)}' for _, _, du, _ in s.disks]
    mnt_w = max(cell_len(m) for m in [d[0] for d in s.disks] + ['挂载点'])
    used_w = max(cell_len(u) for u in used + ['已用/总量'])
    bw = max(inner - mnt_w - used_w - 7 - 9 * 2 - 12 - 1, 5)  # 7: 百分比, 9*2: 读写列, 12: 4 个列间隙 (内边距+分隔符)
    for (mnt, dev, du, rw), u in zip(s.disks, used):
        t.add_row(mnt, line(bar(du.percent, bw), f' {du.percent:4.1f}%'), u,
                  *((fmt_rate(rw[0]), fmt_rate(rw[1])) if rw else ('-', '-')))
    return t, len(s.disks) + 2


def net_body(s, inner):
    t = table(('网卡', {}), ('下行', RATE), ('上行', RATE), ('累计收', R), ('累计发', R))
    for nic, rate, rx, tx in s.nics:
        t.add_row(nic, *((fmt_rate(rate[0]), fmt_rate(rate[1])) if rate else ('-', '-')), fmt_bytes(rx), fmt_bytes(tx))
    lines = [t]
    if s.nics_hidden:
        lines.append(line((f'另有 {s.nics_hidden} 个网卡未显示 (按当前速率取前 {MAX_NICS})', 'dim')))
    for key, arrow in (('rx', '↓'), ('tx', '↑')):
        peak = max(s.hist[key])
        label = f' 峰值 {fmt_rate(peak)}'
        w = inner - 2 - cell_len(label)
        lines.append(line(f'{arrow} ', spark(s.hist[key], w, peak, 'cyan'), (label, 'dim')))
    return Group(*lines), len(lines) + len(s.nics) + 1


def gpu_body(s, gpu):
    if gpu.status:
        return line((gpu.status, 'dim')), 1
    t = table(('GPU', {}), ('型号', {}), ('利用率', {}), ('显存', {}), ('温度', R), ('功耗', R))
    for g in s.gpus:
        mp = g.mem_used / g.mem_total * 100
        util = line(bar(g.util, 20), f' {g.util:3d}%') if g.util is not None else '-'
        power = '-' if g.power is None else f'{g.power / 1000:.0f}W' + (
            '' if g.power_limit is None else f'/{g.power_limit / 1000:.0f}W')
        t.add_row(str(g.idx), g.name, util,
                  line(bar(mp, 20), f' {fmt_bytes(g.mem_used)}/{fmt_bytes(g.mem_total)}'),
                  '-' if g.temp is None else f'{g.temp}°C', power)
    return t, len(s.gpus) + 2


def top_cpu_body(s, k):
    t = table(('PID', R), ('用户', {}), ('CPU%', R), ('内存', R), ('命令', CMD))
    for pid, cpu, rss, _, user, cmd in s.top_cpu[:k]:
        t.add_row(str(pid), user, Text(f'{cpu:.1f}', style=level(cpu)), fmt_bytes(rss), cmd)
    return t, min(k, len(s.top_cpu)) + 2


def top_mem_body(s, cg, k):
    t = table(('PID', R), ('用户', {}), ('内存', R), ('MEM%', R), ('命令', CMD))
    for pid, _, rss, _, user, cmd in s.top_mem[:k]:
        pct = rss / cg.mem_limit * 100
        t.add_row(str(pid), user, fmt_bytes(rss), Text(f'{pct:.1f}', style=level(pct)), cmd)
    return t, min(k, len(s.top_mem)) + 2


def top_gpu_body(s):
    if not s.gpu_procs:
        return line(('当前无 GPU 进程', 'dim')), 1
    t = table(('GPU', R), ('PID', R), ('用户', {}), ('显存', R), ('命令', CMD))
    for gi, pid, used, user, cmd in s.gpu_procs:
        t.add_row(str(gi), str(pid), user, '不可用' if used is None else fmt_bytes(used), cmd)
    return t, len(s.gpu_procs) + 2


def row(width, left, right):
    """左右两个面板等高并排, 终端过窄时上下堆叠. 返回 [(区块, 行数)]."""
    (lb, ln, lt), (rb, rn, rt) = left, right
    if width < NARROW:
        return [(panel(lb, lt), ln + 2), (panel(rb, rt), rn + 2)]
    lay = Layout()
    lay.split_row(Layout(panel(lb, lt)), Layout(panel(rb, rt)))
    return [(lay, max(ln, rn) + 2)]


def render(s, cg, gpu, width, height, interval, cost):
    """各区块尺寸固定, 用 Layout 直接切分屏幕, 省去 rich 对嵌套表格的反复测量."""
    inner = (width if width < NARROW else width // 2) - 4
    head = Table.grid(expand=True)
    head.add_column()
    head.add_column(justify='right')
    head.add_row(line((socket.gethostname(), 'bold'), ('  运行 ', 'dim'), fmt_dur(s.uptime)),
                 line((time.strftime('%Y-%m-%d %H:%M:%S'), 'bold')))
    scope = line(('统计范围  ', 'dim'), cg.scope)
    cpu_title = 'CPU' + ('  (各核心为物理核视角)' if cg.cpu_dir else '')
    gb, gn = gpu_body(s, gpu)
    parts = [(head, 1), (scope, 1)]
    parts += row(width, (*cpu_body(s, cg, inner), cpu_title), (*mem_body(s, cg, inner), '内存'))
    parts += row(width, (*disk_body(s, inner), '磁盘  (IO 为设备级)'), (*net_body(s, inner), '网络'))
    parts.append((panel(gb, 'GPU'), gn + 2))
    tail = []
    if gpu.handles:
        tb, tn = top_gpu_body(s)
        tail.append((panel(tb, 'GPU Top (按显存)'), tn + 2))
    tail.append((line((f'刷新 {interval:g}s  |  上轮采样+渲染 {cost * 1000:.0f}ms  |  monitor 自身 CPU '
                       f'{s.self_cpu:.1f}%  RSS {fmt_bytes(s.self_rss)}  |  按 q 退出', 'dim')), 1))
    # 终端高度不足时减少 CPU/内存 TopN 的行数 (标题显示实际行数), 每个面板另占边框与表头 4 行
    rest = height - sum(n for _, n in parts + tail)
    k = max(1, min(len(s.top_cpu), rest - 4 if width >= NARROW else (rest - 8) // 2))
    parts += row(width, (*top_cpu_body(s, k), f'CPU Top{k}  (100%=1核)'),
                 (*top_mem_body(s, cg, k), f'内存 Top{k}  (MEM% 相对内存限额)'))
    root = Layout()
    root.split_column(*(Layout(r, size=n) for r, n in parts + tail))
    return root


# ---------------------------------------------------------------- 主循环

def main():
    ap = argparse.ArgumentParser(description='轻量终端资源监控')
    ap.add_argument('-i', '--interval', type=float, default=1.0, help='刷新间隔秒数 (默认 1)')
    ap.add_argument('-n', '--top', type=int, default=8, help='TopN 进程数量 (默认 8)')
    args = ap.parse_args()
    if args.interval <= 0 or args.top <= 0:
        ap.error('--interval 与 --top 必须为正数')
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        sys.exit('monitor 需要在交互式终端中运行')

    cg = detect_cgroup()
    gpu = Gpu()
    sampler = Sampler(cg, gpu, args.top)
    console = Console()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    cost = 0.0
    try:
        with Live(Text('正在采样...'), console=console, screen=True, auto_refresh=False) as live:
            deadline = time.monotonic() + args.interval
            while True:
                wait = deadline - time.monotonic()
                if wait > 0 and select.select([fd], [], [], wait)[0]:
                    if b'q' in os.read(fd, 64).lower():
                        break
                    continue
                t0 = time.perf_counter()
                s = sampler.sample()
                live.update(render(s, cg, gpu, console.width, console.height, args.interval, cost), refresh=True)
                cost = time.perf_counter() - t0
                deadline = max(deadline + args.interval, time.monotonic())
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == '__main__':
    main()
