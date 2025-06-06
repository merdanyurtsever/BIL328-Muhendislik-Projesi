#!/bin/bash
# check_env.sh - Basic environment check

echo "Environment Check"
echo "================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Python version: $(python --version 2>&1)"
echo "PyTorch installed: $(pip list | grep torch || echo 'Not found')"
echo "Directory: $(pwd)"
echo "Files:"
ls -la | head

echo -e "\nRunning basic Python test..."
python -c "import sys; print(f'Python version: {sys.version}')"

echo -e "\nRunning PyTorch test..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo -e "\nMemory usage:"
free -h

echo -e "\nDisk usage:"
df -h | head

echo -e "\nRunning processes:"
ps aux | head

echo -e "\nCheck completed."
