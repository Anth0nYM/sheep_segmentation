import cv2
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Any, Optional, Callable, Union


class SheepDataset(Dataset):
    def __init__(self,
                 dir_path: str,
                 transform: Optional[Callable] = None
                 ) -> None:

        self._dir_path = dir_path
        self._img_paths = glob(f'{dir_path}/images/*.jpg')
        self._mask_paths = glob(f'{dir_path}/masks/*.png')
        self._transform = transform

        if len(self._img_paths) != len(self._mask_paths):
            raise ValueError("Lenght mismatch between images x masks")

    def __len__(self) -> int:
        return len(self._img_paths)

    def __getitem__(self,
                    idx: int
                    ) -> tuple[Any, Any]:

        image_path = self._img_paths[idx]
        image_name = image_path[15:-4]
        mask_path = self._mask_paths[idx]
        mask_name = mask_path[14:-12]

        if image_name != mask_name:
            raise ValueError(f'Name mismatch: {image_name}, {mask_name}')

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self._transform:
            transformed = self._transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        mask = mask.float() / 255.0  # type: ignore
        mask = mask.unsqueeze(0)  # type: ignore
        return image, mask


class CustomDataLoader:
    def __init__(self,
                 batch_size: int,
                 shuffle: bool,
                 img_size: int,
                 subset_size: Optional[int] = None,
                 seed: int = 0,
                 augment: bool = False,
                 ) -> None:

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._img_size = img_size
        self._subset_size = subset_size
        self._seed = seed
        self._augment = augment
        self._dir_path = 'dataset'

        self._transform = self._get_transforms()
        self._dataset = SheepDataset(
            dir_path=self._dir_path,
            transform=self._transform['train']
        )
        self._train_set, self._val_set, self._test_set = self._split_dataset()

    def _get_transforms(self,
                        proba: float = 0.5
                        ) -> dict[str, A.Compose]:

        train_transform = (
            A.Compose([
                # TODO crop not resize
                A.Resize(height=self._img_size, width=self._img_size),
                # TODO modify in place
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]) if self._augment else A.Compose([
                A.Resize(height=self._img_size, width=self._img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        )

        val_test_transform = A.Compose([
            A.Resize(height=self._img_size, width=self._img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        return {
            'train': train_transform,
            'val': val_test_transform,
            'test': val_test_transform
        }

    def _split_dataset(self) -> list[Subset]:
        total_size = len(self._dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(self._seed)
        return random_split(
            self._dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

    def _create_dataloader(self,
                           subset: Dataset
                           ) -> Union[DataLoader, tqdm]:

        if self._subset_size:
            subset = Subset(subset, list(range(self._subset_size)))

        dataloader = DataLoader(
            dataset=subset,
            batch_size=self._batch_size,
            shuffle=self._shuffle
        )

        return dataloader

    def get_train_dataloader(self) -> Union[DataLoader, tqdm]:
        return self._create_dataloader(self._train_set)

    def get_val_dataloader(self) -> Union[DataLoader, tqdm]:
        return self._create_dataloader(self._val_set)

    def get_test_dataloader(self) -> Union[DataLoader, tqdm]:
        return self._create_dataloader(self._test_set)
