# CRP Quality Classification System

A CLIP-based classification system for identifying the aging stage of Citri Reticulatae Pericarpium (CRP) using consumer-grade camera images.

## Project Structure

```text
├── clip_resnet/
│   ├── dataload.py
│   ├── model_clip.py
│   └── main_clip.ipynb
├── clip_vit/
│   ├── dataload.py
│   ├── clip_vit.py
│   └── clip_vit_L_14_main.ipynb
├── Original/
│   ├── dataload.py
│   ├── models.py
│   └── main.ipynb
├── app.py
└── README.md
```

## Installation

```bash
pip install streamlit
pip install git+https://github.com/openai/CLIP.git
pip install torch torchvision
pip install Pillow numpy
pip install scikit-learn
```

## How to Run

### Step 1 — Train the Model

Open `clip_resnet/main_clip.ipynb` in Google Colab and run all cells. After training, download the exported model file:

```text
best_model_seed_42.pt
```

### Step 2 — Launch the Application

Place `best_model_seed_42.pt` in the same directory as `app.py` and run:

```bash
streamlit run app.py
```

### Step 3 — Perform Inference

Upload three images of the same CRP sample:

* Full RGB image
* Black background image
* White background image

The application will return the predicted aging stage along with confidence scores for each class.

## Example Output

```text
Predicted Class : 2800 days
Confidence      : 84.3%

Class Scores:
190 days  - 4.1%
560 days  - 6.2%
2800 days - 84.3%
3300 days - 5.4%
```

## Experiments

### Original Baseline

The `Original/` directory contains our reimplementation of the original paper's architecture. The model uses a three-stream ResNet50 design with channel-level feature interaction between full, black-background, and white-background image streams. This implementation served as the baseline for all subsequent experiments.

### CLIP Vision Transformer

We replaced the ResNet50 backbone with CLIP Vision Transformers and evaluated multiple variants including ViT-B/32, ViT-B/16, and ViT-L/14. A lightweight linear classification head was trained on top of frozen CLIP features, with and without dropout regularization.

The best-performing configuration was ViT-L/14 with a linear layer and dropout, achieving approximately 73% test accuracy.

Performance was limited by the relatively small dataset size of approximately 400 images. Although the CLIP encoder remained frozen, a noticeable train-test gap persisted. Additionally, Vision Transformers process images as patch sequences, which reduced the effectiveness of the channel interaction mechanism used in the original three-stream architecture.

### CLIP ResNet50 (Final Model)

The final approach replaces the original ResNet50 backbone with CLIP RN50 while preserving the three-stream interaction architecture proposed in the paper.

Unlike Vision Transformers, CLIP ResNet50 maintains spatial feature maps throughout the network, making it naturally compatible with channel-level interaction between image streams. This model achieved the highest performance across all experiments.

The improved results can be attributed to:

* Preservation of spatial feature maps, which better support stream interaction
* Complementary information provided by the full, black-background, and white-background image streams
* Stronger pretrained visual representations learned by CLIP compared to standard ImageNet-pretrained ResNet50

The final model combines CLIP RN50 with the original three-stream interaction framework and produced the best overall classification performance.

## Dependencies

* torch
* torchvision
* clip
* streamlit
* Pillow
* numpy
* scikit-learn

## Link to Dataset
https://drive.google.com/drive/folders/1zz13y12S0q_XlhAWfdOyEu8Pu1PKj1mF
