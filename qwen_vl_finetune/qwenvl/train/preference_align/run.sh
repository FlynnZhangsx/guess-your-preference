#!/bin/bash
# AI Image Studio — 一键启动脚本
cd "$(dirname "$0")"
VENV=/home/coder/project/data/mllm/duomotai/mllm/bin/python
LOG=/tmp/webui.log

echo "Starting AI Image Studio..."
nohup $VENV -u api_server.py --port 7860 > $LOG 2>&1 &
PID=$!
echo "PID: $PID"

echo "Waiting for models to load (1-2 min)..."
for i in $(seq 1 12); do
  sleep 15
  STATUS=$(curl -s http://127.0.0.1:7860/api/status 2>/dev/null)
  if echo "$STATUS" | grep -q '"ready":true'; then
    echo "✅ Ready! Access via: /proxy/7860/"
    exit 0
  fi
  echo "  [$i] Loading..."
done
echo "⚠️  Timeout. Check: tail -f $LOG"
