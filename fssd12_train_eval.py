import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.utils.metrics import AUPRO, AUROC
from pytorch_lightning import seed_everything
from torchmetrics import AveragePrecision

from datamodules.fssd12 import FSSD12
from model.supersimplenet import SuperSimpleNet
from train import train


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    prediction = scores >= threshold
    labels = labels.astype(bool)
    tp = np.logical_and(prediction, labels).sum()
    fp = np.logical_and(prediction, ~labels).sum()
    fn = np.logical_and(~prediction, labels).sum()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "accuracy": float((prediction == labels).mean()),
    }


def _box_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    def boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        return [
            tuple(int(v) for v in row[:4])
            for row in stats[1:]
            if int(row[cv2.CC_STAT_AREA]) > 0
        ]

    def iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = first
        bx1, by1, bw, bh = second
        ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - intersection
        return _safe_div(intersection, union)

    predicted = boxes(pred_mask)
    target = boxes(gt_mask)
    matched = 0
    used: set[int] = set()
    overlaps: list[float] = []
    for predicted_box in predicted:
        candidates = [
            (iou(predicted_box, target_box), index)
            for index, target_box in enumerate(target)
            if index not in used
        ]
        if candidates:
            overlap, index = max(candidates)
            if overlap >= 0.5:
                matched += 1
                used.add(index)
                overlaps.append(overlap)
    precision = _safe_div(matched, len(predicted))
    recall = _safe_div(matched, len(target))
    return {
        "box_precision": precision,
        "box_recall": recall,
        "box_f1": _safe_div(2 * precision * recall, precision + recall),
        "box_ap50": precision * recall,
        "box_mean_iou": float(np.mean(overlaps)) if overlaps else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: SuperSimpleNet,
    datamodule: FSSD12,
    device: str,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    image_scores, image_labels, pixel_scores, pixel_labels = [], [], [], []
    box_values = []
    for batch in datamodule.test_dataloader():
        anomaly_map, anomaly_score = model(batch["image"].to(device))
        image_scores.extend(torch.sigmoid(anomaly_score).cpu().numpy())
        image_labels.extend(batch["label"].numpy())
        maps = torch.sigmoid(anomaly_map).cpu().numpy()
        masks = batch["mask"].numpy()
        for score_map, mask in zip(maps, masks):
            score_map = score_map.squeeze()
            mask = mask.squeeze() > 0.5
            pixel_scores.extend(score_map.ravel())
            pixel_labels.extend(mask.ravel())
            box_values.append(_box_metrics(score_map >= threshold, mask))

    image_scores = np.asarray(image_scores)
    image_labels = np.asarray(image_labels)
    pixel_scores = np.asarray(pixel_scores)
    pixel_labels = np.asarray(pixel_labels)
    image_metrics = _binary_metrics(image_scores, image_labels, threshold)
    pixel_metrics = _binary_metrics(pixel_scores, pixel_labels, threshold)
    result = {
        "image_auroc": AUROC()(torch.tensor(image_scores), torch.tensor(image_labels)).item(),
        "image_ap": AveragePrecision(num_classes=1)(
            torch.tensor(image_scores), torch.tensor(image_labels)
        ).item(),
        "pixel_auroc": AUROC()(torch.tensor(pixel_scores), torch.tensor(pixel_labels)).item(),
        "pixel_ap": AveragePrecision(num_classes=1)(
            torch.tensor(pixel_scores), torch.tensor(pixel_labels)
        ).item(),
        "pixel_aupro": AUPRO()(
            torch.tensor(pixel_scores).reshape(-1, 256, 256),
            torch.tensor(pixel_labels).reshape(-1, 256, 256),
        ).item(),
        **image_metrics,
        "pixel_precision": pixel_metrics["precision"],
        "pixel_recall": pixel_metrics["recall"],
        "pixel_f1": pixel_metrics["f1"],
        "pixel_iou": _safe_div(
            pixel_metrics["f1"],
            2 - pixel_metrics["f1"],
        ),
    }
    for key in box_values[0]:
        result[key] = float(np.mean([item[key] for item in box_values]))
    return result


@torch.no_grad()
def export_predictions(
    model: SuperSimpleNet,
    datamodule: FSSD12,
    category: str,
    device: str,
    threshold: float,
    rows: list[dict[str, object]],
) -> None:
    model.eval()
    for batch in datamodule.test_dataloader():
        _, anomaly_score = model(batch["image"].to(device))
        scores = torch.sigmoid(anomaly_score).reshape(-1).cpu().numpy()
        labels = batch["label"].numpy()
        for image_path, label, score in zip(batch["image_path"], labels, scores):
            predicted_label = int(score >= threshold)
            rows.append(
                {
                    "category": category,
                    "image_path": image_path,
                    "true_label": "anomalous" if label else "normal",
                    "predicted_label": (
                        "anomalous" if predicted_label else "normal"
                    ),
                    "anomaly_score": float(score),
                    "threshold": threshold,
                    "correct": predicted_label == int(label),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/sunlong/wmh/data/FSSD-12"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("./fssd12_results"))
    parser.add_argument("--seed", type=int, default=456654)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Anomaly score threshold for binary predictions.",
    )
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    categories = sorted(path.name for path in args.data_root.iterdir() if path.is_dir())
    if args.categories:
        unknown = sorted(set(args.categories) - set(categories))
        if unknown:
            parser.error(f"Unknown categories: {', '.join(unknown)}")
        categories = args.categories
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    prediction_rows = []

    for category in categories:
        seed_everything(args.seed, workers=True)
        config = {
            "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"],
            "patch_size": 3, "noise": True, "perlin": True, "no_anomaly": "empty",
            "bad": True, "overlap": False, "adapt_cls_feat": False,
            "noise_std": 0.015, "perlin_thr": 0.6, "image_size": (256, 256),
            "seg_lr": 0.0002, "dec_lr": 0.0002, "adapt_lr": 0.0001,
            "gamma": 0.4, "stop_grad": False,
            "epochs": args.epochs,
        }
        datamodule = FSSD12(
            args.data_root, category, train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size, num_workers=0, seed=args.seed, flips=True
        )
        datamodule.setup()
        model = SuperSimpleNet(image_size=(256, 256), config=config)
        checkpoint = args.output / "checkpoints" / category / "weights.pt"
        if args.export_predictions and checkpoint.is_file():
            model.load_model(checkpoint)
            model.to(device)
            export_predictions(
                model, datamodule, category, device, args.threshold, prediction_rows
            )
            continue
        train(
            model, args.epochs, datamodule, device,
            {"I-AUROC": AUROC(), "AP-det": AveragePrecision(num_classes=1)},
            {"P-AUROC": AUROC(), "AUPRO": AUPRO(), "AP-loc": AveragePrecision(num_classes=1)},
            clip_grad=True, eval_step_size=max(args.epochs, 1),
        )
        result = {
            "category": category,
            **evaluate(model, datamodule, device, args.threshold),
        }
        rows.append(result)
        model.save_model(args.output / "checkpoints" / category)
        with (args.output / f"{category}.json").open("w") as file:
            json.dump(result, file, indent=2)

    if rows:
        with (args.output / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    if args.export_predictions:
        with (args.output / "predictions.csv").open("w", newline="") as file:
            fieldnames = [
                "category",
                "image_path",
                "true_label",
                "predicted_label",
                "anomaly_score",
                "threshold",
                "correct",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)


if __name__ == "__main__":
    main()
