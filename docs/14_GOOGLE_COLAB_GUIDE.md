# Google Colab Integration Guide

This guide details how to run the TCC Deepfake Detection platform in Google Colab.

## 1. Setup

### 1.1. Prepare Google Drive
1. Create a folder named `TCC` in your Google Drive root.
2. Upload the entire project content to this folder.
   - Ideally, use `git clone` inside Colab or upload the zip file and extract it.

### 1.2. Directory Structure
Ensure your Drive looks like this:
```
/content/drive/MyDrive/TCC/
├── app/
├── data/
│   ├── raw/
│   │   ├── real/
│   │   └── fake/
│   └── processed/
├── notebooks/
├── requirements.txt
└── ...
```

## 2. Notebooks Overview

We have prepared 3 notebooks to handle different stages of the pipeline:

### 2.1. [01_Setup_and_Extraction.ipynb](../01_Setup_and_Extraction.ipynb)
- **Purpose**: Installs dependencies and extracts features from raw audio files.
- **Input**: Audio files in `data/raw/real` and `data/raw/fake`.
- **Output**: `real_features.joblib` and `fake_features.joblib` in `data/processed/`.
- **Runtime**: Standard CPU or GPU (GPU recommended for faster extraction with some libraries).

### 2.2. [02_Training.ipynb](../02_Training.ipynb)
- **Purpose**: Trains the Deepfake Detection model.
- **Input**: Extracted features from Notebook 01.
- **Output**: Trained model saved in `app/models/`.
- **Runtime**: **GPU Required**. Enable it in `Runtime > Change runtime type > T4 GPU`.

### 2.3. [03_Inference_and_Demo.ipynb](../03_Inference_and_Demo.ipynb)
- **Purpose**: Runs the Gradio Interface for testing and demonstration.
- **Features**: 
    - Real-time recording or file upload.
    - Deepfake probability score.
    - Feature visualization.
- **Runtime**: CPU or GPU.
- **Access**: Creates a public shareable link (e.g., `https://xxxx.gradio.live`).

## 3. Common Issues & Solutions

### 3.1. Path Issues
- Always run the setup cell that mounts Drive and sets `PROJECT_PATH`.
- If `ModuleNotFoundError` occurs, ensure `sys.path.append(PROJECT_PATH)` was executed.

### 3.2. GPU Availability
- Run `!nvidia-smi` to check if a GPU is attached.
- Some advanced architectures (like AASIST) might require specific CUDA versions. The default Colab environment is usually sufficient.

### 3.3. Memory
- If the session crashes during feature extraction, try reducing the `batch_size` or processing files in smaller chunks.

## 4. Running the Demo
1. Open `03_Inference_and_Demo.ipynb`.
2. Run all cells.
3. Click the public link provided in the output of the last cell (e.g., `Running on public URL: https://...`).
