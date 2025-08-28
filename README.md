# 1. En tu terminal
python -m venv venv          # Crea el env
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

pip install torch torchvision torchaudio  # CUDA o CPU según tu tarjeta
pip install opencv-python
pip install ultralytics==8.0.0  # YOLOv8 incluye YOLOv5, pero se puede usar solo v5