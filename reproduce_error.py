import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
from app.domain.models.architectures.efficientnet_lstm import create_efficientnet_lstm_model

def test_model_creation():
    print("Testing create_efficientnet_lstm_model...")
    input_shape = (100, 80) # 2D input (Time, Freq)
    num_classes = 2
    
    try:
        model = create_efficientnet_lstm_model(
            input_shape=input_shape,
            num_classes=num_classes
        )
        print("Model created successfully.")
        model.summary()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_creation()
