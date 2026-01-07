# Google Colab Setup for ONN-TLM
# 1. Upload 'onn_colab_package.zip' to the Colab runtime (drag & drop to the left file panel)
# 2. Run this cell to setup the environment

!unzip -o onn_colab_package.zip

import os
import sys

# Add src to python path
sys.path.append(os.path.abspath("src"))

# Install standard dependencies if missing (Colab usually has them)
!pip install -q torch numpy matplotlib

print("✅ Setup complete! You can now run the training scripts.")
print("Example:")
print("!python3 scripts/train_oscillator.py --epochs 500 --embed-dim 512 --save")
