#!/usr/bin/env bash
# 一键启动终端资源监控. 用法: ./monitor.sh [-i 刷新间隔秒数] [-n TopN数量]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python3 -c 'import importlib.util as u, sys; sys.exit(not all(u.find_spec(m) for m in ("psutil", "rich", "pynvml")))'; then
    echo "缺少依赖, 正在安装: psutil rich nvidia-ml-py"
    python3 -m pip install psutil rich nvidia-ml-py
fi
exec python3 "$DIR/monitor.py" "$@"
