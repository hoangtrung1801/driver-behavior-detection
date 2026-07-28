import torch

from cdira.models.cdira import CDIRA, RoutingPolicy


def test_full_training_forward_shapes() -> None:
    model = CDIRA(num_classes=10, num_domains=30, pretrained=False)
    output = model.forward_train(torch.randn(2, 3, 224, 224))
    assert output.global_logits.shape == (2, 10)
    assert output.fused_logits.shape == (2, 10)
    assert output.routing_logits.shape == (2,)
    assert output.domain_logits.shape == (2, 30)
    assert output.saliency.shape == (2, 7, 7)


def test_all_easy_inference_skips_roi_branch() -> None:
    model = CDIRA(num_classes=10, num_domains=3, pretrained=False)
    output = model.predict(torch.randn(2, 3, 224, 224), RoutingPolicy.NONE, 0.9)
    assert not output.roi_mask.any()
    assert torch.all(output.topk_indices == -1)


def test_mixed_inference_routes_only_selected_rows() -> None:
    model = CDIRA(num_classes=10, num_domains=3, pretrained=False)
    output = model.predict(torch.randn(2, 3, 224, 224), RoutingPolicy.HEAD, 0.0)
    assert output.roi_mask.all()
    assert (output.topk_indices >= 0).all()
