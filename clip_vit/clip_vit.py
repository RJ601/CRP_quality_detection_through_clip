# models.py

import torch
import torch.nn as nn
import clip

class CLIPZeroShotModel(nn.Module):
    def __init__(self, class_names: list, clip_model_name: str = "ViT-L/14"):
        super(CLIPZeroShotModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

        # Load pretrained CLIP
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)

        # Freeze all CLIP weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Trainable linear head on top of frozen CLIP image features
        self.classifier = nn.Linear(768, len(class_names)).to(self.device)

    def forward(self, images: torch.Tensor):
        images = images.to(self.device)

        # Frozen CLIP image encoder
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Trainable head — this is what actually gets trained on your data
        logits = self.classifier(image_features.float())
        return logits