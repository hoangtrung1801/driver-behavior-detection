from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cdira.config import ExperimentConfig, load_config
from cdira.data.dataset import build_transform
from cdira.models.cdira import CDIRA, RoutingPolicy

CLASS_NAMES = {f"c{index}": f"Driver behavior c{index}" for index in range(10)}


def aggregate_frame_predictions(probabilities: np.ndarray) -> dict[str, Any]:
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must have shape [frames, classes]")
    frame_predictions = probabilities.argmax(axis=1).astype(int).tolist()
    mean_probabilities = probabilities.mean(axis=0)
    class_index = int(mean_probabilities.argmax())
    return {
        "class_index": class_index,
        "confidence": float(mean_probabilities[class_index]),
        "probabilities": mean_probabilities,
        "frame_predictions": frame_predictions,
    }


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _selected_domains(checkpoint: Path) -> int:
    metrics_path = checkpoint.parent.parent / "metrics" / "full.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        value = payload.get("selected_domains")
        if isinstance(value, int) and value > 0:
            return value
    return 25


def load_model(
    checkpoint: Path,
    config_path: Path | None = None,
    device: torch.device | None = None,
) -> tuple[CDIRA, ExperimentConfig, torch.device]:
    checkpoint = checkpoint.expanduser().resolve()
    resolved_config = checkpoint.parent.parent / "config.resolved.yaml"
    config = load_config(resolved_config if resolved_config.exists() else (config_path or Path("configs/standard.yaml")))
    selected_device = device or _default_device()
    model = CDIRA(
        num_classes=config.model.num_classes,
        num_domains=_selected_domains(checkpoint),
        top_k=config.model.top_k,
        global_hidden=config.model.global_hidden,
        roi_hidden=config.model.roi_hidden,
        fused_hidden=config.model.fused_hidden,
        routing_hidden=config.model.routing_hidden,
        domain_hidden=config.model.domain_hidden,
        grl_strength=config.model.grl_strength,
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(selected_device)
    return model, config, selected_device


def predict_image(
    model: CDIRA,
    image: Image.Image,
    config: ExperimentConfig,
    device: torch.device,
    policy: RoutingPolicy = RoutingPolicy.HEAD,
    threshold: float | None = None,
) -> dict[str, Any]:
    transform = build_transform(
        False, config.data.image_size, False, config.data.brightness
    )
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    routing_threshold = threshold if threshold is not None else config.routing.threshold
    with torch.no_grad():
        output = model.predict(tensor, policy, routing_threshold)
    probabilities = output.logits.softmax(dim=1)[0].detach().cpu()
    prediction = int(probabilities.argmax())
    return {
        "class_id": f"c{prediction}",
        "class_name": CLASS_NAMES.get(f"c{prediction}", f"Class c{prediction}"),
        "confidence": float(probabilities[prediction]),
        "probabilities": probabilities.numpy(),
        "routing_probability": float(output.routing_probability[0].detach().cpu()),
        "global_confidence": float(output.global_confidence[0].detach().cpu()),
        "roi_used": bool(output.roi_mask[0].detach().cpu()),
        "saliency": output.saliency[0].detach().cpu(),
    }


def sample_video_frames(
    video_bytes: bytes, max_frames: int = 32
) -> tuple[list[Image.Image], list[int], float]:
    import cv2

    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    with tempfile.NamedTemporaryFile(suffix=".video") as temporary:
        temporary.write(video_bytes)
        temporary.flush()
        capture = cv2.VideoCapture(temporary.name)
        if not capture.isOpened():
            raise ValueError("Could not open the uploaded video")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0:
            capture.release()
            raise ValueError("The uploaded video contains no readable frames")
        count = min(frame_count, max_frames)
        indices = np.linspace(0, frame_count - 1, count, dtype=int).tolist()
        frames: list[Image.Image] = []
        read_indices: list[int] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = capture.read()
            if success:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
                read_indices.append(index)
        capture.release()
    if not frames:
        raise ValueError("Could not decode any frames from the uploaded video")
    return frames, read_indices, fps


def predict_video(
    model: CDIRA,
    video_bytes: bytes,
    config: ExperimentConfig,
    device: torch.device,
    policy: RoutingPolicy = RoutingPolicy.HEAD,
    threshold: float | None = None,
    max_frames: int = 32,
    batch_size: int = 8,
) -> dict[str, Any]:
    frames, frame_indices, fps = sample_video_frames(video_bytes, max_frames)
    transform = build_transform(
        False, config.data.image_size, False, config.data.brightness
    )
    routing_threshold = threshold if threshold is not None else config.routing.threshold
    probabilities: list[np.ndarray] = []
    routing: list[np.ndarray] = []
    roi_flags: list[np.ndarray] = []
    saliencies: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            batch = torch.stack([transform(frame) for frame in frames[start : start + batch_size]])
            output = model.predict(batch.to(device), policy, routing_threshold)
            probabilities.append(output.logits.softmax(dim=1).cpu().numpy())
            routing.append(output.routing_probability.cpu().numpy())
            roi_flags.append(output.roi_mask.cpu().numpy())
            saliencies.extend(output.saliency.cpu())
    probability_array = np.concatenate(probabilities, axis=0)
    aggregate = aggregate_frame_predictions(probability_array)
    frame_confidences = probability_array.max(axis=1)
    representative_index = int(frame_confidences.argmax())
    class_id = f"c{aggregate['class_index']}"
    aggregate.update(
        {
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, f"Class {class_id}"),
            "frame_indices": frame_indices,
            "fps": fps,
            "frame_confidences": frame_confidences,
            "routing_probabilities": np.concatenate(routing),
            "roi_usage": float(np.concatenate(roi_flags).mean()),
            "frames": frames,
            "representative_saliency": saliencies[representative_index],
        }
    )
    return aggregate


def make_saliency_overlay(image: Image.Image, saliency: torch.Tensor) -> Image.Image:
    base_image = image.convert("RGB")
    saliency_array = saliency.detach().float().cpu().numpy()
    saliency_array = np.squeeze(saliency_array)
    minimum = float(saliency_array.min())
    maximum = float(saliency_array.max())
    if maximum > minimum:
        saliency_array = (saliency_array - minimum) / (maximum - minimum)
    else:
        saliency_array = np.zeros_like(saliency_array)
    heatmap = Image.fromarray((saliency_array * 255).astype(np.uint8)).resize(
        base_image.size, Image.Resampling.BILINEAR
    )
    heat = np.asarray(heatmap, dtype=np.float32) / 255.0
    base = np.asarray(base_image, dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    alpha = (heat * 0.55)[..., None]
    blended = base * (1.0 - alpha) + red * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="C-DIRA Driver Behavior", page_icon="🚗", layout="wide")
    st.title("C-DIRA Driver Behavior Detection")
    st.caption("Upload a State Farm driver image and inspect the trained C-DIRA prediction.")

    with st.sidebar:
        st.header("Model")
        checkpoint_text = st.text_input(
            "Checkpoint",
            "artifacts/standard/run/checkpoints/cdira.pt",
        )
        policy_name = st.selectbox("Routing policy", ["head", "confidence"])
        threshold = st.slider("Routing threshold", 0.1, 0.99, 0.9, 0.01)
        input_mode = st.radio("Input", ["Image", "Video"])

    checkpoint = Path(checkpoint_text).expanduser()
    if not checkpoint.exists():
        st.error(f"Checkpoint not found: {checkpoint}")
        st.info("Train a model first or select another checkpoint path in the sidebar.")
        return

    @st.cache_resource(show_spinner="Loading C-DIRA checkpoint...")
    def cached_model(path: str) -> tuple[CDIRA, ExperimentConfig, torch.device]:
        return load_model(Path(path))

    try:
        model, config, device = cached_model(str(checkpoint.resolve()))
    except (OSError, RuntimeError, ValueError) as exc:
        st.exception(exc)
        return

    st.sidebar.success(f"Loaded on {device}")
    if input_mode == "Video":
        uploaded_video = st.file_uploader(
            "Upload a driver video", type=["mp4", "mov", "avi", "m4v"]
        )
        if uploaded_video is None:
            st.info("Choose a video to begin.")
            return
        video_bytes = uploaded_video.getvalue()
        st.video(video_bytes)
        try:
            video_result = predict_video(
                model,
                video_bytes,
                config,
                device,
                RoutingPolicy(policy_name),
                threshold,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            st.exception(exc)
            return
        left, right = st.columns(2)
        with left:
            st.subheader(f"Video prediction: {video_result['class_id']}")
            st.metric("Behavior", video_result["class_name"])
            st.metric("Aggregated confidence", f"{video_result['confidence']:.2%}")
            st.metric("ROI usage", f"{video_result['roi_usage']:.2%}")
            st.write(f"Analyzed frames: {len(video_result['frames'])}")
        with right:
            representative = video_result["frames"][int(video_result["frame_confidences"].argmax())]
            st.image(representative, caption="Most confident sampled frame", use_container_width=True)
        timeline = pd.DataFrame(
            {
                "frame": video_result["frame_indices"],
                "prediction": [
                    f"c{index}" for index in video_result["frame_predictions"]
                ],
                "confidence": video_result["frame_confidences"],
                "routing_probability": video_result["routing_probabilities"],
            }
        )
        st.subheader("Frame timeline")
        st.dataframe(timeline, use_container_width=True, hide_index=True)
        st.subheader("Representative ROI saliency")
        st.image(
            make_saliency_overlay(representative, video_result["representative_saliency"]),
            caption="Red areas indicate high backbone saliency.",
            use_container_width=True,
        )
        return

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Choose a driver image to begin.")
        return

    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    result = predict_image(
        model,
        image,
        config,
        device,
        RoutingPolicy(policy_name),
        threshold,
    )
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with right:
        st.subheader(f"Prediction: {result['class_id']}")
        st.metric("Class", result["class_name"])
        st.metric("Confidence", f"{result['confidence']:.2%}")
        st.metric("ROI refinement", "Used" if result["roi_used"] else "Not used")
        st.write(f"Routing probability: {result['routing_probability']:.2%}")
        st.write(f"Global confidence: {result['global_confidence']:.2%}")

    probabilities = pd.DataFrame(
        {
            "class": [f"c{index}" for index in range(config.model.num_classes)],
            "probability": result["probabilities"],
        }
    ).set_index("class")
    st.subheader("Class probabilities")
    st.bar_chart(probabilities)
    st.subheader("ROI saliency")
    st.image(
        make_saliency_overlay(image, result["saliency"]),
        caption="Red areas indicate high backbone saliency; ROI routing may still be disabled.",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
