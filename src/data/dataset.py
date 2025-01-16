from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
from .preproc import preproc as pp


class SheepsDataset(Dataset):
    def __init__(self,
                 path: str,
                 to_augment: bool,
                 image_size: tuple[int, int],
                 idxs=None,
                 pre_load: str = "data.csv"
                 ) -> None:

        self.__path = path
        self.__to_augment = to_augment
        self.__pre_load_path = os.path.join(self.__path, pre_load)
        self.__image_size = image_size

        self.__dataset_name = self.__path.split("/")[-1]
        self.__weights_dir = os.path.join(self.__path, "weights.xlsx")
        self.__weights = pd.read_excel(self.__weights_dir, index_col=0)
        self.__images_dir = os.path.join(self.__path, "images")
        self.__masks_dir = os.path.join(self.__path, "masks")

        if not os.path.exists(self.__pre_load_path):
            self.__create_dataframe(self.__dataset_name)
        else:
            self.__data = pd.read_csv(self.__pre_load_path)

        if idxs is not None:
            self.__data = self.__data.iloc[idxs].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self,
                    idx: int
                    ) -> tuple:

        row = self.__data.iloc[idx]
        image_path = row["image_path"]
        mask_path = row["mask_path"]

        id = int(row["animal_id"])
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        weight = row["weight"]

        image, mask, weight = pp(img=image,
                                 mask=mask,
                                 weight=weight,
                                 to_augment=self.__to_augment,
                                 img_size=self.__image_size)

        return id, image, mask, weight

    def get_ids(self) -> list:
        return self.__data["animal_id"].tolist()

    def __assert_lens(self, img_folder: str, mask_folder: str) -> bool:
        len_img_folder = len(next(os.walk(img_folder))[1])
        len_mask_folder = len(next(os.walk(mask_folder))[1])
        len_weights = len(self.__weights)

        if len_weights == len_img_folder == len_mask_folder:
            return True
        else:
            print(f"Number of sheeps in weights: {len_weights}")
            print(f"Number of sheeps in images: {len_img_folder}")
            print(f"Number of sheeps in masks: {len_mask_folder}")

            raise ValueError("Count mismatch")

    def __create_dataframe(self, dataset_name) -> None:
        if dataset_name not in ["will", "alci"]:
            raise ValueError(
                "Dataset not found, the dataset root name must be will or alci"
            )

        if dataset_name == "will":
            self.__get_will()

        elif dataset_name == "alci":
            self.__get_alci()

    def __get_will(self) -> None:
        data = []
        if self.__assert_lens(self.__images_dir, self.__masks_dir):
            for animal_id in os.listdir(self.__masks_dir):
                mask_folder = os.path.join(self.__masks_dir,
                                           animal_id,
                                           "mask_rgb")
                image_folder = os.path.join(self.__images_dir, animal_id)

                for mask_file in os.listdir(mask_folder):
                    mask_path = os.path.join(mask_folder, mask_file)
                    image_name = mask_file.replace("_mask", "")
                    image_path = os.path.join(image_folder, image_name)

                    if os.path.exists(image_path):
                        weight = self.__weights.loc[int(animal_id), "Peso"]
                        data.append({
                            "animal_id": int(animal_id),
                            "image_path": image_path,
                            "mask_path": mask_path,
                            "weight": weight
                        })

        self.__data = pd.DataFrame(data)
        self.__data.to_csv(self.__pre_load_path, index=False)

        return None

    def __get_alci(self) -> None:
        print("Not implemented yet")
        print("Returning Will data")
        self.__get_will()
