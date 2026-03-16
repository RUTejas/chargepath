#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "=================================================="
echo "  EV Charging AI  |  Q1 Research Dashboard"
echo "=================================================="
python3 --version || { echo "Python 3.8+ required"; exit 1; }
[ ! -d "venv" ] && python3 -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet flask numpy pandas scikit-learn matplotlib scipy
python3 -c "import torch" 2>/dev/null || \
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
echo "Open: http://localhost:5000"
python3 app.py
