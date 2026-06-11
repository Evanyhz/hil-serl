#!/usr/bin/env python3
"""Inspect robosuite_door reward-classifier data.

Reads ./classifier_data/*.pkl, validates transition structure, prints dataset
statistics, exports wrist/side/merged PNGs, and optionally previews samples.

command example:
python3 -B inspect_classifier_data.py --export-images --max-export 50

"""

from __future__ import annotations

import argparse
import pickle as pkl
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "classifier_data"
DEFAULT_EXPORT_DIR = SCRIPT_DIR / "classifier_data_preview"

OBS_KEYS = {"wrist", "side", "state"}
TRANSITION_KEYS = {
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "masks",
    "dones",
}
FORBIDDEN_KEYS = {
    "images",
    "hinge_qpos",
    "handle_qpos",
    "handle_pos",
    "door_pos",
    "door_pose",
    "handle_pose",
    "success",
    "succeed",
    "is_success",
    "true_contact_point",
    "true_contact_force",
    "handle_to_eef",
    "door_to_eef",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect robosuite_door classifier_data pkl files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--max-export", type=int, default=50, help="Maximum samples per label to export.")
    parser.add_argument("--stride", type=int, default=1, help="Export every Nth sample.")
    parser.add_argument("--export-images", action="store_true", help="Export wrist/side/merged PNG previews.")
    parser.add_argument("--show", action="store_true", help="Interactively show samples with OpenCV.")
    parser.add_argument("--no-validate", action="store_true", help="Skip strict transition/key validation.")
    return parser.parse_args()


def label_from_path(path: Path) -> str:
    name = path.name.lower()
    if "success" in name:
        return "success"
    if "failure" in name or "fail" in name:
        return "failure"
    return "unknown"


def iter_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("*.pkl"))
    return [path for path in paths if label_from_path(path) != "unknown"]


def assert_no_forbidden_keys(value: Any, where: str = "transition") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key = str(key)
            assert key not in FORBIDDEN_KEYS, f"forbidden key at {where}.{key}"
            assert_no_forbidden_keys(child, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_forbidden_keys(child, f"{where}[{index}]")


def validate_transition(trans: Mapping[str, Any], path: Path, index: int) -> None:
    prefix = f"{path.name}[{index}]"
    assert set(trans.keys()) == TRANSITION_KEYS, f"{prefix}: bad transition keys {sorted(trans.keys())}"

    obs = trans["observations"]
    next_obs = trans["next_observations"]
    assert set(obs.keys()) == OBS_KEYS, f"{prefix}: bad obs keys {sorted(obs.keys())}"
    assert set(next_obs.keys()) == OBS_KEYS, f"{prefix}: bad next_obs keys {sorted(next_obs.keys())}"

    action = np.asarray(trans["actions"])
    assert action.shape == (7,), f"{prefix}: bad action shape {action.shape}"

    for obs_name, obs_value in (("observations", obs), ("next_observations", next_obs)):
        for image_key in ("wrist", "side"):
            image = np.asarray(obs_value[image_key])
            assert image.ndim in (3, 4), f"{prefix}: {obs_name}.{image_key} bad ndim {image.ndim}"
            if image.ndim == 4:
                assert image.shape[0] == 1, f"{prefix}: {obs_name}.{image_key} leading dim must be 1, got {image.shape}"
                image = image[0]
            assert image.ndim == 3 and image.shape[-1] == 3, f"{prefix}: {obs_name}.{image_key} bad shape {image.shape}"
            assert image.dtype == np.uint8, f"{prefix}: {obs_name}.{image_key} dtype should be uint8, got {image.dtype}"

        state = np.asarray(obs_value["state"])
        assert state.ndim in (1, 2), f"{prefix}: {obs_name}.state bad shape {state.shape}"

    assert_no_forbidden_keys(trans, prefix)


def squeeze_image(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image[..., :3])


def update_stats(stats: dict[str, Any], trans: Mapping[str, Any]) -> None:
    obs = trans["observations"]
    action = np.asarray(trans["actions"], dtype=np.float32)
    state = np.asarray(obs["state"], dtype=np.float32)
    wrist = squeeze_image(obs["wrist"])
    side = squeeze_image(obs["side"])

    stats["action_norms"].append(float(np.linalg.norm(action)))
    stats["state_shapes"].append(tuple(state.shape))
    stats["wrist_shapes"].append(tuple(wrist.shape))
    stats["side_shapes"].append(tuple(side.shape))
    stats["wrist_min"].append(int(wrist.min()))
    stats["wrist_max"].append(int(wrist.max()))
    stats["side_min"].append(int(side.min()))
    stats["side_max"].append(int(side.max()))


def load_dataset(data_dir: Path, validate: bool) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    paths = iter_paths(data_dir)
    if not paths:
        raise FileNotFoundError(f"No success/failure pkl files found in {data_dir}")

    for path in paths:
        label = label_from_path(path)
        data = pkl.load(path.open("rb"))
        if not isinstance(data, list):
            raise TypeError(f"{path} should contain a list of transitions, got {type(data)}")

        for index, trans in enumerate(data):
            if validate:
                validate_transition(trans, path, index)
            samples.append(
                {
                    "label": label,
                    "path": path,
                    "index": index,
                    "transition": trans,
                }
            )
    return samples


def print_summary(samples: list[dict[str, Any]]) -> None:
    label_counts = Counter(sample["label"] for sample in samples)
    file_counts = Counter(str(sample["path"].name) for sample in samples)

    stats_by_label: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {
            "action_norms": [],
            "state_shapes": [],
            "wrist_shapes": [],
            "side_shapes": [],
            "wrist_min": [],
            "wrist_max": [],
            "side_min": [],
            "side_max": [],
        }
    )

    for sample in samples:
        update_stats(stats_by_label[sample["label"]], sample["transition"])

    print("Dataset summary")
    print("===============")
    print(f"total samples: {len(samples)}")
    print(f"success: {label_counts.get('success', 0)}")
    print(f"failure: {label_counts.get('failure', 0)}")
    print("")

    print("files:")
    for name, count in sorted(file_counts.items()):
        print(f"  {name}: {count}")
    print("")

    for label in ("success", "failure"):
        stats = stats_by_label[label]
        if not stats["action_norms"]:
            continue

        print(f"{label} details")
        print("-" * (len(label) + 8))
        print(f"state shapes: {Counter(stats['state_shapes'])}")
        print(f"wrist shapes: {Counter(stats['wrist_shapes'])}")
        print(f"side shapes: {Counter(stats['side_shapes'])}")
        print(
            "action norm: "
            f"min={np.min(stats['action_norms']):.4f}, "
            f"mean={np.mean(stats['action_norms']):.4f}, "
            f"max={np.max(stats['action_norms']):.4f}"
        )
        print(f"wrist pixel range: min={min(stats['wrist_min'])}, max={max(stats['wrist_max'])}")
        print(f"side pixel range: min={min(stats['side_min'])}, max={max(stats['side_max'])}")
        print("")


def sample_for_export(samples: list[dict[str, Any]], label: str, max_export: int, stride: int) -> Iterable[dict[str, Any]]:
    label_samples = [sample for sample in samples if sample["label"] == label]
    stride = max(1, stride)
    exported = 0

    for sample in label_samples[::stride]:
        if exported >= max_export:
            break
        exported += 1
        yield sample


def export_images(samples: list[dict[str, Any]], export_dir: Path, max_export: int, stride: int) -> None:
    import cv2

    export_dir.mkdir(parents=True, exist_ok=True)

    for label in ("success", "failure"):
        label_dir = export_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for out_index, sample in enumerate(sample_for_export(samples, label, max_export, stride)):
            obs = sample["transition"]["observations"]
            wrist = squeeze_image(obs["wrist"])
            side = squeeze_image(obs["side"])
            merged = np.concatenate([wrist, side], axis=1)

            stem = f"{label}_{out_index:04d}_src{sample['path'].stem}_idx{sample['index']:06d}"
            cv2.imwrite(str(label_dir / f"{stem}_wrist.png"), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(label_dir / f"{stem}_side.png"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(label_dir / f"{stem}_merged.png"), cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))

    print(f"exported previews to: {export_dir}")


def show_samples(samples: list[dict[str, Any]]) -> None:
    import cv2

    if not samples:
        return

    index = 0
    cv2.namedWindow("classifier_data_inspector", cv2.WINDOW_NORMAL)

    while True:
        sample = samples[index]
        obs = sample["transition"]["observations"]
        wrist = squeeze_image(obs["wrist"])
        side = squeeze_image(obs["side"])
        merged = cv2.cvtColor(np.concatenate([wrist, side], axis=1), cv2.COLOR_RGB2BGR)
        merged = cv2.resize(merged, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

        text = f"{index + 1}/{len(samples)} label={sample['label']} file={sample['path'].name} idx={sample['index']}"
        cv2.putText(merged, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(merged, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("classifier_data_inspector", merged)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("a"), ord("h")):
            index = max(0, index - 1)
        elif key in (ord("d"), ord("l"), ord(" ")):
            index = min(len(samples) - 1, index + 1)

    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    export_dir = args.export_dir.expanduser()

    samples = load_dataset(data_dir, validate=not args.no_validate)
    print_summary(samples)

    if args.export_images:
        export_images(samples, export_dir, args.max_export, args.stride)

    if args.show:
        show_samples(samples)


if __name__ == "__main__":
    main()