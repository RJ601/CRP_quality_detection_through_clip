<div align="center">

# 🍊 CRP Quality Detection Using CLIP

**Citri Reticulatae Pericarpium · Vintage & Authenticity Classification**

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991?style=flat-square&logo=openai&logoColor=white)](https://github.com/openai/CLIP)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)

</div>

---

## What is this?

Citri Reticulatae Pericarpium (CRP) — dried citrus peel — is a prized edible and medicinal ingredient whose value depends on **origin and aging**. Premium aged CRP from Xinhui can cost ¥3300/kg, while counterfeit products masquerade as the real thing. Distinguishing them visually is hard even for experts.

This project builds a **deep learning classifier** that takes three smartphone photos of a CRP piece and predicts its vintage/authenticity category in real time — no lab equipment needed.

We extend the multi-stream architecture of [Wu et al. (PLOS One, 2026)](https://doi.org/10.1371/journal.pone.0340161) by replacing their ResNet50 backbone with **OpenAI's CLIP-RN50**, adding auxiliary heads, fixing reproducibility, and deploying a Streamlit frontend.

---

## Resnet50 (Original Baseline)

The `Original/` directory contains our reimplementation of the original paper's architecture. The model uses a three-stream ResNet50 design with channel-level feature interaction between full, black-background, and white-background image streams. This implementation served as the baseline for all subsequent experiments.

## CLIP Vision Transformer (Experimental)

We replaced the ResNet50 backbone with CLIP Vision Transformers and evaluated multiple variants including ViT-B/32, ViT-B/16, and ViT-L/14. A lightweight linear classification head was trained on top of frozen CLIP features, with and without dropout regularization.

The best-performing configuration was ViT-L/14 with a linear layer and dropout, achieving approximately 73% test accuracy.

Performance was limited by the relatively small dataset size of approximately 400 images. Although the CLIP encoder remained frozen, a noticeable train-test gap persisted. Additionally, Vision Transformers process images as patch sequences, which reduced the effectiveness of the channel interaction mechanism used in the original three-stream architecture.

---

## CLIP-RN50 (Main Architecture)

Three parallel CLIP-RN50 branches process complementary views of the same CRP specimen:

<img width="1200" height="940" alt="clip_architecture" src="https://github.com/user-attachments/assets/9b7188d7-420b-49bd-bb4b-1234c21cd1d1" />

**Key design choices:**

- **Backbone**: CLIP-RN50 (pretrained on 400M image-text pairs). Uses CLIP's 3-conv stem + avgpool instead of standard single conv + maxpool.
- **Freezing**: All layers frozen except Layer4, reducers, auxiliary heads, and main classifier.
- **Feature Interaction** at Layer2: 10% of Full branch channels injected into Black & White branches (forward), then 5% from each local branch injected back into Full (reverse).
- **Seed 142** fixed for channel selection → fully reproducible results.
- **Auxiliary heads**: Per-branch classifiers. Total loss = main + aux_full + aux_black + aux_white.
- **Dropout 0.5** on all classifiers and the 2048→512 dimension reducer.

---

## Dataset

Images sourced from the official repository of Wu et al. (iPhone captures only).

| Label | Origin | Type | Price (CNY/kg) | Samples |
|---|---|---|---|---|
| 190 | Wuzhou, Guangxi | Counterfeit | 190 | 120 |
| 560 | Yunfu, Guangdong | Counterfeit | 560 | 105 |
| 2800 | Xinhui, Guangdong | Genuine · 10yr+ | 2800 | 84 |
| 3300 | Xinhui, Guangdong | Genuine · 15yr+ | 3300 | 90 |

Each specimen provides **three images**: a full image (whole peel, exocarp side up), a black patch (exocarp close-up), and a white patch (albedo/inner surface).

Domain shift for Vivo and Xiaomi was **simulated** via brightness, sharpness, contrast, and noise augmentation since those device images were unavailable.

<img width="2480" height="900" alt="Peel_Images_Diff_Devices" src="https://github.com/user-attachments/assets/afc8300e-dc6c-4a1e-a185-8b148eee3037" />

---

## Results at a Glance

| Model | Test Acc (iPhone) | Vivo Acc | Xiaomi Acc |
|---|---|---|---|
| **CLIP-Interact (ours)** | **98.33%** | **92.0%** | 75.8% |
| Baseline (ResNet50) | 97.08% | 78.6% | **86.8%** |

> CLIP-RN50 achieves **+13.4% improvement on Vivo** simulated images — reflecting stronger robustness to mild domain shift from CLIP's large-scale pretraining.

---

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

---

## Setup

```bash
# 1. Create and activate environment
conda create -n crp python=3.9
conda activate crp

# 2. Install dependencies
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install streamlit Pillow numpy

# 3. Add model weights
# Place best_model_seed_242.pt in the project root
```

---

## Running the Frontend

```bash
streamlit run app.py
```

The app asks for three images:

| Upload slot | What to provide |
|---|---|
| **Full image** | Whole CRP piece, exocarp side facing up |
| **Black patch** | Close-up of the dark outer exocarp surface |
| **White patch** | Close-up of the inner albedo (flipped side) |

Once all three are uploaded, click **Run Inference** to get the predicted class and per-class confidence scores.

> The model loads once on startup and stays cached — subsequent predictions are fast.

---

## Training

Open `main_clip.ipynb` in Jupyter/Colab. The notebook covers:

- Dataset loading (full / black / white streams)
- Training loop with auxiliary loss
- Validation and early stopping
- Cross-device evaluation on simulated Vivo/Xiaomi images
- Confusion matrix and accuracy plots

Training config used: Adam optimizer · lr=1e-4 · cosine annealing · 30 epochs · early stopping.

---

## What We Changed from the Baseline

| | Baseline (Wu et al.) | Ours |
|---|---|---|
| Backbone | ResNet50 | **CLIP-RN50** |
| Reproducibility | ❌ Random channel selection unseeded | ✅ Fixed seed 142 |
| Overfitting control | None in repo | ✅ Dropout 0.5 + auxiliary heads |
| Faster R-CNN | In paper (not in repo) | Not needed (patches pre-provided) |
| MAML | In paper | Dropped (memory constraints) |
| Frontend | None | ✅ Streamlit app |

---

## Known Limitations

- **Xiaomi performance** is consistently lower for CLIP-RN50 (75.8%) vs baseline (86.8%). Our hypothesis: CLIP's texture-sensitive filters are disproportionately hurt by the aggressive Xiaomi augmentation (sharpness 1.8, noise std=8). Exact cause is an open question — real Xiaomi images would help confirm.
- Only 4 CRP classes. Generalization to other price points or fruit varieties untested.
- Domain shift results are based on simulated images, not actual multi-device captures.

---

## Future Work

The reasons for this CLIP-RN50’s exceptionally degraded performance on the xiaomi might be the drastic augmentations but when compared to performance of the original network, the difference is still not understood and could be studied using explainable AI to improve the network. 

The networks could also be tested and compared on actual instead of simulated images of CRP taken from different specification cameras to give results more aligned with real-life use-cases.

In addition to this, we could make this research more generalizable by adding diverse data of different classes, hence extending its ability to detect quality for a wide variety of fruits and vegetables. By using a larger dataset, we can also see the CLIP ViT’s full capacity as it performs the best with big amounts of data, so we don’t limit its classification ability that the dataset used in this study did.

---

## Link to Dataset
https://drive.google.com/drive/folders/1zz13y12S0q_XlhAWfdOyEu8Pu1PKj1mF

---

## References

Wu Z, Wang T, Mao Z, Huang L, Chen J, Yang X (2026). *A convenient method for the accurate identification of Citri Reticulatae Pericarpium using image and multi-stream.* PLoS One 21(2): e0340161. https://doi.org/10.1371/journal.pone.0340161

Radford A et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.

---

<div align="center">

Made by **Javeria Rahman (23i-0137) & Shireen Fatima (23i-0130)** 
BS Artificial Intelligence · NUCES Islamabad · Instructor: Sir Ishtiaq

</div>
