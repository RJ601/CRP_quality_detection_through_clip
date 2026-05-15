import torch
import torch.nn as nn
import clip

class CLIPCombinedInteractModel(nn.Module):
    def __init__(self, num_classes, interaction_layer="layer2", interaction_ratio=0.1, freeze_before_layer4=True):
        super(CLIPCombinedInteractModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.interaction_layer_name = interaction_layer
        self.interaction_ratio = interaction_ratio
        self.layer_names = ["layer1", "layer2", "layer3", "layer4"]
        self.pool = nn.AdaptiveAvgPool2d(1).to(self.device)

        # auxiliary classifiers
        self.aux_classifier_full = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes)).to(self.device)
        self.aux_classifier_black = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes)).to(self.device)
        self.aux_classifier_white = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes)).to(self.device)

        def create_clip_visual():
            # CLIP ka RN50 visual encoder load karo
            clip_model, _ = clip.load("RN50", device="cpu")
            v = clip_model.visual  # yeh modified ResNet hai

            # Freeze karo early layers (layer3/layer4 trainable rakhte hain)
            for name, param in v.named_parameters():
                if freeze_before_layer4:
                    if "layer4" not in name:
                        param.requires_grad = False
                        
            return v.to(self.device)

        self.full_image_model  = create_clip_visual()
        self.black_image_model = create_clip_visual()
        self.white_image_model = create_clip_visual()

        
        # Ek baar calculate karo, hamesha same rahega
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            x = self._clip_stem(self.full_image_model, dummy)
            for ln in self.layer_names:
                x = getattr(self.full_image_model, ln)(x)
                if ln == self.interaction_layer_name:
                    C = x.shape[1]  # actual channels
                    break
        g = torch.Generator().manual_seed(142)
        self.register_buffer('full_channels',  torch.randperm(C, generator=g)[:int(C * 0.10)])
        self.register_buffer('black_channels', torch.randperm(C, generator=g)[:int(C * 0.05)])
        self.register_buffer('white_channels', torch.randperm(C, generator=g)[:int(C * 0.05)])

        # Yeh same hai original code se — 2048 -> 512
        def make_reducer():
            return nn.Sequential(
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5)
            ).to(self.device)

        self.reduce_dim_full  = make_reducer()
        self.reduce_dim_black = make_reducer()
        self.reduce_dim_white = make_reducer()

        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(512 * 3, num_classes)
                            ).to(self.device)

    def _clip_stem(self, model, x):
        """
        CLIP RN50 ka stem original ResNet se thoda alag hai —
        3 conv layers hain (conv1/conv2/conv3) instead of ek.
        Original code mein sirf conv1+bn1+relu+maxpool tha.
        """
        # CLIP uses a 3-conv stem
        x = model.relu1(model.bn1(model.conv1(x)))
        x = model.relu2(model.bn2(model.conv2(x)))
        x = model.relu3(model.bn3(model.conv3(x)))
        x = model.avgpool(x)  # CLIP uses avgpool here, not maxpool
        return x

    def _feature_interaction(self, full, black, white):
        """
        通道级别的特征交互
        full: 给出10%的通道给black和white
        black/white: 各自给出5%的通道给full
        """
        if full is None or black is None or white is None:
            return full, black, white       
        device = full.device
        black = black.to(device)
        white = white.to(device)        

        # 创建交互后的特征图
        new_full = full.clone()
        new_black = black.clone()
        new_white = white.clone()

        new_black[:, self.full_channels]  = full[:, self.full_channels]
        new_white[:, self.full_channels]  = full[:, self.full_channels]
        new_full[:, self.black_channels]  = black[:, self.black_channels]
        new_full[:, self.white_channels]  = white[:, self.white_channels]

        return new_full, new_black, new_white

    def forward(self, full_image, black_image, white_image):
        full_image  = full_image.to(self.device)
        black_image = black_image.to(self.device)
        white_image = white_image.to(self.device)

        # Stem: CLIP ka 3-conv stem use karo (original mein 1-conv tha)
        f = self._clip_stem(self.full_image_model,  full_image)
        b = self._clip_stem(self.black_image_model, black_image)
        w = self._clip_stem(self.white_image_model, white_image)

        # Yeh loop bilkul same hai original se
        for layer_name in self.layer_names:
            full_layer  = getattr(self.full_image_model,  layer_name)
            black_layer = getattr(self.black_image_model, layer_name)
            white_layer = getattr(self.white_image_model, layer_name)

            f = full_layer(f)
            b = black_layer(b)
            w = white_layer(w)

            if layer_name == self.interaction_layer_name:
                f, b, w = self._feature_interaction(f, b, w)

        # CLIP mein attnpool hai — hum use SKIP karte hain, apna avgpool lagao
        f = torch.flatten(self.pool(f), 1)  # (B, 2048)
        b = torch.flatten(self.pool(b), 1)
        w = torch.flatten(self.pool(w), 1)

        f = self.reduce_dim_full(f)
        b = self.reduce_dim_black(b)
        w = self.reduce_dim_white(w)

        aux_f = self.aux_classifier_full(f)
        aux_b = self.aux_classifier_black(b)
        aux_w = self.aux_classifier_white(w)

        combined = torch.cat([f, b, w], dim=1)  # (B, 1536)
        return self.classifier(combined), aux_f, aux_b, aux_w