import cv2
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import MatLike, Any,  Optional, Callable


class Data(Dataset):
    def __init__(self,
                 dir: str,
                 transform: Optional[Callable] = None) -> None:

        self._dir = dir
        self._img_paths = sorted(glob(pathname=f'{dir}/images/*.jpg'))
        self._mask_paths = sorted(glob(pathname=f'{dir}/masks/*.png'))
        self._transform = transform

    def __len__(self) -> int:
        return len(self._img_paths)

    def __getitem__(self, idx: int) -> tuple[MatLike | Any, MatLike | Any]:
        image_path = self._img_paths[idx]
        image_name = image_path[15:-4]
        mask_path = self._mask_paths[idx]
        mask_name = mask_path[14:-12]

        if image_name != mask_name:
            raise Exception(f'Name mismatch: {image_name}, {mask_name}')

        image = cv2.imread(filename=image_path)
        mask = cv2.imread(filename=mask_path)

        if self._transform:
            transformed = self._transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask


class Dataloader:
    def __init__(self,
                 batch_size: int,
                 shuffle: bool,
                 size: int,
                 subset: int = 0,
                 seed: int = 0,
                 description: bool = False) -> None:

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._size = size
        self._subset = subset
        self._seed = seed
        self._description = description
        self._dir = 'dataset'
        self.transform = self.compose()
        self.dataset = Data(dir=self._dir, transform=self.transform['train'])
        self.train_set, self.val_set, self.test_set = self.split_dataset()

    def compose(self) -> dict[str, A.Compose]:
        trans_train = A.Compose(transforms=[
            A.Resize(height=self._size, width=self._size),
            ToTensorV2()
        ])
        trans_test = A.Compose([
            A.Resize(height=self._size, width=self._size),
            ToTensorV2()
        ])
        return {
            'train': trans_train,
            'val': trans_test,
            'test': trans_test
        }

    def split_dataset(self) -> tuple[Subset, Subset, Subset]:
        total_size = len(self.dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(self._seed)
        train_set, val_set, test_set = random_split(
            dataset=self.dataset,
            lengths=[train_size, val_size, test_size],
            generator=generator
        )
        return train_set, val_set, test_set

    def get_dataloader(self, subset: Dataset) -> DataLoader:
        if self._subset:
            subset = Subset(dataset=subset, indices=list(range(self._subset)))
        dataloader = DataLoader(
            dataset=subset,
            batch_size=self._batch_size,
            shuffle=self._shuffle
        )

        return tqdm(dataloader) if self._description else dataloader

    def get_train_dataloader(self) -> DataLoader:
        return self.get_dataloader(subset=self.train_set)

    def get_val_dataloader(self) -> DataLoader:
        return self.get_dataloader(subset=self.val_set)

    def get_test_dataloader(self) -> DataLoader:
        return self.get_dataloader(subset=self.test_set)
