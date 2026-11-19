# Road Accident Detection System

A deep learning model that classifies CCTV frames as **Accident** or **Non Accident**, with **Grad-CAM** heatmaps that show which part of the image drove the prediction. Built with PyTorch and trained on Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_92YVwBCvXWzvWVlF_0oX35IxcE0bySw?usp=sharing)

![Grad-CAM example: predicted Accident (0.634)](assets/gradcam_example.png)

## Highlights

- **EfficientNet-B3** backbone (pretrained, via `timm`) for feature extraction
- **CBAM attention** (channel + spatial) added on top of the backbone features
- **Focal loss** with label smoothing and class weights to handle class imbalance
- **Weighted random sampling** so each class is seen evenly during training
- **Grad-CAM** visualisation for explainable predictions
- **Streamlit demo app** for uploading an image or video and viewing predictions

## Results

Evaluated on the held-out test split (100 images) after 20 epochs of training.

| Metric | Value |
|---|---|
| Test accuracy | **98%** |
| Final training accuracy | 98.7% |

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Accident | 1.00 | 0.96 | 0.98 | 47 |
| Non Accident | 0.96 | 1.00 | 0.98 | 53 |

Confusion matrix (rows = actual, columns = predicted):

|  | Pred. Accident | Pred. Non Accident |
|---|---|---|
| **Actual Accident** | 45 | 2 |
| **Actual Non Accident** | 0 | 53 |

The model missed 2 accident frames and raised no false alarms on this test set.

## Model Architecture

```
Input (300 x 300 RGB)
   -> EfficientNet-B3 backbone (1536 feature channels)
   -> CBAM (Channel Attention -> Spatial Attention)
   -> Adaptive Average Pooling
   -> Dropout (0.4)
   -> Linear layer (2 classes)
```

## Training Setup

| Setting | Value |
|---|---|
| Input size | 300 x 300 |
| Batch size | 32 |
| Epochs | 20 |
| Optimizer | AdamW (lr = 3e-4) |
| Loss | Focal loss (gamma = 2.0, label smoothing = 0.1, class weights) |
| Sampling | `WeightedRandomSampler` (inverse class frequency) |
| Augmentation | Random horizontal flip, random rotation (10 degrees) |
| Normalisation | ImageNet mean and std |

## Dataset

[Accident Detection from CCTV Footage](https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage) from Kaggle (`ckay16/accident-detection-from-cctv-footage`).

Expected folder layout:

```
data/
  train/
    Accident/
    Non Accident/
  test/
    Accident/
    Non Accident/
```

## Getting Started

### Run on Google Colab (recommended)

1. Click the **Open in Colab** badge above.
2. Set the runtime to **GPU** (Runtime -> Change runtime type).
3. Run all cells. The dataset is downloaded automatically with `kagglehub`.

### Run locally

```bash
git clone https://github.com/Swarna-Kamal/road-accident-detection.git
cd road-accident-detection

pip install torch torchvision timm seaborn scikit-learn matplotlib numpy opencv-python pillow kagglehub streamlit
```

Then open the notebook and run the cells in order. A CUDA GPU is strongly recommended for training.

## Grad-CAM Explainability

Grad-CAM highlights the regions of a frame that most influenced the prediction, for example the vehicles involved in a collision. In the notebook, `show_gradcam(image_path, model, device, class_names)` overlays the heatmap on the image and shows the predicted class with its confidence.

## Streamlit Demo

The notebook also writes an `app.py` Streamlit app that lets you:

- Upload an **image** or a **video** (`mp4`, `avi`)
- Adjust the **confidence threshold**
- View the prediction side by side with the **Grad-CAM** overlay
- For videos, process every 5th frame and list the frames flagged as accidents

Run it with:

```bash
streamlit run app.py
```

On Colab, expose it with `localtunnel`:

```bash
npm install -g localtunnel
streamlit run app.py & lt --port 8501
```

## Tech Stack

Python, PyTorch, torchvision, timm, scikit-learn, OpenCV, Matplotlib, Seaborn, Streamlit, Google Colab

## Limitations and Future Work

- The test set is small (100 images), so the reported accuracy should be treated as indicative rather than definitive.
- Frames are classified one at a time, so the model does not use motion or temporal information across a video.
- Future work: temporal models (for example LSTM or 3D CNN) for video, a larger and more varied dataset, and testing under different lighting and weather conditions.

## Author

**Swarna Kamal Panguluri**
[GitHub](https://github.com/Swarna-Kamal) | [LinkedIn](https://www.linkedin.com/in/swarna-kamal-panguluri/) | [LeetCode](https://leetcode.com/u/Swarna_Kamal/)
