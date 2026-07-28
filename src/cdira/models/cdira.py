from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from cdira.models.grl import gradient_reverse
from cdira.models.roi import topk_roi_pool


class RoutingPolicy(str, Enum):
    HEAD = "head"
    CONFIDENCE = "confidence"
    NONE = "none"


@dataclass(frozen=True)
class TrainOutput:
    global_logits: Tensor
    fused_logits: Tensor
    routing_logits: Tensor
    domain_logits: Tensor
    saliency: Tensor
    topk_indices: Tensor


@dataclass(frozen=True)
class InferenceOutput:
    logits: Tensor
    global_logits: Tensor
    routing_probability: Tensor
    global_confidence: Tensor
    roi_mask: Tensor
    saliency: Tensor
    topk_indices: Tensor


class CDIRA(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        top_k: int = 5,
        global_hidden: int = 256,
        roi_hidden: int = 512,
        fused_hidden: int = 512,
        routing_hidden: int = 128,
        domain_hidden: int = 256,
        grl_strength: float = 1.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features
        self.feature_dim = 576
        self.top_k = top_k
        self.grl_strength = grl_strength
        self.global_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, global_hidden),
            nn.ReLU(),
            nn.Linear(global_hidden, num_classes),
        )
        self.roi_refinement = nn.Sequential(
            nn.Linear(self.feature_dim, roi_hidden), nn.ReLU()
        )
        self.fused_classifier = nn.Sequential(
            nn.Linear(self.feature_dim + roi_hidden, fused_hidden),
            nn.ReLU(),
            nn.Linear(fused_hidden, num_classes),
        )
        self.routing_head = nn.Sequential(
            nn.Linear(self.feature_dim, routing_hidden),
            nn.ReLU(),
            nn.Linear(routing_hidden, 1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, num_domains),
        )

    def _features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        feature_map = self.backbone(images)
        global_feature = torch.nn.functional.adaptive_avg_pool2d(
            feature_map, 1
        ).flatten(1)
        return feature_map, global_feature

    def _fused_logits(
        self, global_feature: Tensor, feature_map: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        roi = topk_roi_pool(feature_map, self.top_k)
        refined = self.roi_refinement(roi.pooled)
        fused = self.fused_classifier(torch.cat([global_feature, refined], dim=1))
        return fused, roi.saliency, roi.indices

    def forward_train(self, images: Tensor) -> TrainOutput:
        feature_map, global_feature = self._features(images)
        global_logits = self.global_classifier(global_feature)
        fused_logits, saliency, topk_indices = self._fused_logits(
            global_feature, feature_map
        )
        routing_logits = self.routing_head(global_feature).squeeze(-1)
        domain_logits = self.domain_classifier(
            gradient_reverse(global_feature, self.grl_strength)
        )
        return TrainOutput(
            global_logits,
            fused_logits,
            routing_logits,
            domain_logits,
            saliency,
            topk_indices,
        )

    def predict(
        self, images: Tensor, policy: RoutingPolicy, threshold: float
    ) -> InferenceOutput:
        feature_map, global_feature = self._features(images)
        global_logits = self.global_classifier(global_feature)
        probabilities = global_logits.softmax(dim=1)
        global_confidence = probabilities.max(dim=1).values
        routing_probability = self.routing_head(global_feature).squeeze(-1).sigmoid()
        if policy is RoutingPolicy.HEAD:
            roi_mask = routing_probability >= threshold
        elif policy is RoutingPolicy.CONFIDENCE:
            roi_mask = global_confidence < threshold
        else:
            roi_mask = torch.zeros_like(global_confidence, dtype=torch.bool)
        logits = global_logits.clone()
        saliency = torch.zeros_like(feature_map[:, 0])
        topk_indices = torch.full(
            (images.shape[0], self.top_k), -1, dtype=torch.long, device=images.device
        )
        if roi_mask.any():
            fused, selected_saliency, selected_indices = self._fused_logits(
                global_feature[roi_mask], feature_map[roi_mask]
            )
            logits[roi_mask] = fused
            saliency[roi_mask] = selected_saliency
            topk_indices[roi_mask] = selected_indices
        return InferenceOutput(
            logits,
            global_logits,
            routing_probability,
            global_confidence,
            roi_mask,
            saliency,
            topk_indices,
        )


class MobileNetBaseline(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        network = mobilenet_v3_small(weights=weights)
        self.features = network.features
        self.classifier = nn.Linear(576, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        feature_map = self.features(images)
        features = torch.nn.functional.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        return self.classifier(features)  # type: ignore[no-any-return]
