#!/bin/bash
cd /workspace
rm -f DONE
python3 -u align_judge.py > /workspace/run.log 2>&1
echo "EXIT_CODE=$?" >> /workspace/run.log
touch /workspace/DONE
