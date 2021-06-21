import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import resnet

from models.cbam import CBAM


def _weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class My_ResNet(resnet.ResNet):
    def __init__(
        self,
        block,
        layers,
        num_classes=8,
        zero_init_last_bn=True,
        use_cbam=False,
        **kwargs
    ):
        super(My_ResNet, self).__init__(block, layers, num_classes, **kwargs)

        # Additional Modules
        self.fc = None
        self.cbam = (
            CBAM(self.num_features, reduction_ratio=16, bn=True)
            if use_cbam
            else None
        )
        self.alpha = nn.Sequential(
            nn.Linear(self.num_features, 1, bias=False), nn.Sigmoid()
        )
        self.beta = nn.Sequential(
            nn.Linear(self.num_features * 2, 1, bias=False), nn.Sigmoid()
        )
        self.classifier = nn.Linear(self.num_features * 2, num_classes)

        # Weight Initialization
        for m in self.modules():
            _weight_init(m)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, "zero_init_last_bn"):
                    m.zero_init_last_bn()

    def forward_features(self, x):
        features = []
        alphas = []
        for i in range(6):
            f = x[:, :, :, :, i]

            f = super().forward_features(f)

            if self.cbam is not None:
                f = self.cbam(f)

            f = self.global_pool(f)

            features.append(f)
            alphas.append(self.alpha(f))

        # Self-attention Module
        feature_stack = torch.stack(features, dim=2)
        alpha_stack = torch.stack(alphas, dim=2)
        alpha_stack = F.softmax(alpha_stack, dim=2)
        alpha_orig = alpha_stack[:, :, 0]
        alpha_part_max = alpha_stack[:, :, 1:].max(dim=2)[0]
        Fm = feature_stack.mul(alpha_stack).sum(2).div(alpha_stack.sum(2))

        # Relation-attention Module
        betas = []
        for i in range(len(features)):
            features[i] = torch.cat([features[i], Fm], dim=1)
            betas.append(self.beta(features[i]))

        feature_stack = torch.stack(features, dim=2)
        beta_stack = torch.stack(betas, dim=2)
        beta_stack = F.softmax(beta_stack, dim=2)
        output = (
            feature_stack.mul(beta_stack * alpha_stack)
            .sum(2)
            .div((beta_stack * alpha_stack).sum(2))
        )
        output = output.view(output.size(0), -1)

        return output, alpha_part_max, alpha_orig

    def forward(self, x):
        x, alpha_part_max, alpha_orig = self.forward_features(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        pred_score = self.classifier(x)

        return pred_score, alpha_part_max, alpha_orig
