import cv2
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Any, Optional, Callable, Union


class SheepDataset(Dataset):
    def __init__(
        self, dir_path: str, transform: Optional[Callable] = None
    ) -> None:
        self._dir_path = dir_path
        self._img_paths = glob(f'{dir_path}/images/*.jpg')
        self._mask_paths = glob(f'{dir_path}/masks/*.png')
        self._transform = transform

        if len(self._img_paths) != len(self._mask_paths):
            raise ValueError("Lenght mismatch between images x masks")

    def __len__(self) -> int:
        return len(self._img_paths)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        image_path = self._img_paths[idx]
        image_name = image_path[15:-4]
        mask_path = self._mask_paths[idx]
        mask_name = mask_path[14:-12]

        if image_name != mask_name:
            raise ValueError(f'Name mismatch: {image_name}, {mask_name}')

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        if self._transform:
            transformed = self._transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        return image, mask


class CustomDataLoader:
    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        img_size: int,
        subset_size: Optional[int] = None,
        seed: int = 0,
        augment: bool = False,
        show_progress: bool = False
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._img_size = img_size
        self._subset_size = subset_size
        self._seed = seed
        self._augment = augment
        self._show_progress = show_progress
        self._dir_path = 'dataset'

        self._transform = self._get_transforms()
        self._dataset = SheepDataset(
            dir_path=self._dir_path,
            transform=self._transform['train']
        )
        self._train_set, self._val_set, self._test_set = self._split_dataset()

    def _get_transforms(self, proba: float = 0.5) -> dict[str, A.Compose]:
        # TODO ainda não decidi qual exatamente será meu aumento de dados
        # TODO (Quero fazer uma análise de quais são os melhores)
        train_transform = (
            A.Compose([
                A.Resize(height=self._img_size, width=self._img_size),
                # A.RandomCrop(height=self._img_size, width=self._img_size),
                # Color intensity
                # A.CLAHE(p=proba),
                # A.RandomBrightnessContrast(p=proba),
                # A.Blur(p=proba),
                # A.HueSaturationValue(p=proba),

                # Geometric
                # A.HorizontalFlip(p=proba),
                # A.VerticalFlip(p=proba),
                # A.RandomRotate90(p=proba),
                # A.Transpose(p=proba),
                # A.ShiftScaleRotate(
                #     shift_limit=0.0625, scale_limit=0.2, rotate_limit=45,
                # p=proba),
                # Distortion
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=proba),
                A.GridDistortion(p=proba),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=proba),

                ToTensorV2()

            ]) if self._augment else A.Compose([
                A.RandomCrop(height=self._img_size, width=self._img_size),
                ToTensorV2()
            ])
        )

        val_test_transform = A.Compose([
            A.RandomCrop(height=self._img_size, width=self._img_size),
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

    def _create_dataloader(self, subset: Dataset) -> Union[DataLoader, tqdm]:
        if self._subset_size:
            subset = Subset(subset, list(range(self._subset_size)))

        dataloader = DataLoader(
            dataset=subset,
            batch_size=self._batch_size,
            shuffle=self._shuffle
        )

        return tqdm(dataloader) if self._show_progress else dataloader

    def get_train_dataloader(self) -> Union[DataLoader, tqdm]:
        return self._create_dataloader(self._train_set)

    def get_val_dataloader(self) -> Union[DataLoader, tqdm]:
        return self._create_dataloader(self._val_set)

    def get_test_dataloader(self) -> Union[DataLoader, tqdm]:
        return self._create_dataloader(self._test_set)
