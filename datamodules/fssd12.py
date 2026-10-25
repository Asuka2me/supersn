from pathlib import Path

import albumentations as A
import pandas as pd
from anomalib.data.utils import InputNormalizationMethod, Split
from pandas import DataFrame

from datamodules.base import Supervision
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset


class FSSD12Dataset(SSNDataset):
    def __init__(
        self,
        root: Path,
        category: str,
        transform: A.Compose,
        split: Split,
        supervision: Supervision,
        flips: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(
            root=root / category,
            supervision=supervision,
            transform=transform,
            split=split,
            flips=flips,
            normal_flips=False,
            dt=(3, 2) if split == Split.TRAIN else None,
            dilate=7 if split == Split.TRAIN else None,
            debug=debug,
        )
        self.category_root = root / category

    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        image_root = self.category_root / "Images"
        normal_root = self.category_root / "Nd"
        mask_root = self.category_root / "GT"
        train_split = self.split == Split.TRAIN

        def numbered_files(directory: Path, suffix: str) -> list[Path]:
            files = list(directory.glob(suffix))
            return sorted(files, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))

        defect_images = numbered_files(image_root, "*.jpg")
        normal_images = numbered_files(normal_root, "*.jpg")
        if len(defect_images) != 50 or len(normal_images) != 50:
            raise ValueError(
                f"{self.category_root}: expected 50 defect and 50 normal images, "
                f"got {len(defect_images)} and {len(normal_images)}"
            )

        defect_images = defect_images[:30] if train_split else defect_images[30:]
        normal_images = normal_images[:30] if train_split else normal_images[30:]

        normal = pd.DataFrame(
            [
                {
                    "image_path": str(path),
                    "mask_path": "",
                    "label_index": 0,
                    "is_segmented": False,
                }
                for path in normal_images
            ]
        )
        anomalous = pd.DataFrame(
            [
                {
                    "image_path": str(path),
                    "mask_path": str(mask_root / f"{path.stem}.png"),
                    "label_index": 1,
                    "is_segmented": True,
                }
                for path in defect_images
            ]
        )
        missing_masks = [
            row.mask_path
            for row in anomalous.itertuples()
            if not Path(row.mask_path).is_file()
        ]
        if missing_masks:
            raise FileNotFoundError(f"Missing GT masks, for example: {missing_masks[0]}")
        return normal, anomalous


class FSSD12(SSNDataModule):
    def __init__(
        self,
        root: Path | str,
        category: str,
        image_size: tuple[int, int] = (256, 256),
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        flips: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            supervision=Supervision.MIXED_SUPERVISION,
            image_size=image_size,
            normalization=normalization,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            flips=flips,
        )
        self.train_data = FSSD12Dataset(
            Path(root), category, self.transform_train, Split.TRAIN,
            Supervision.MIXED_SUPERVISION, flips=flips
        )
        self.test_data = FSSD12Dataset(
            Path(root), category, self.transform_eval, Split.TEST,
            Supervision.MIXED_SUPERVISION
        )
