import torch
import albumentations as A


def preproc(img, mask, weight, to_augment, img_size) -> tuple:
    if to_augment:
        img, mask = augment(img, mask)

    img, mask, weight = normalize(img=img,
                                  mask=mask,
                                  weight=weight,
                                  img_size=img_size)

    return img, mask, weight


def augment(img, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(0.8, 1.2),
                                   contrast_limit=0,
                                   p=0.5)
    ])

    augmented = transform(image=img, mask=mask)
    img = augmented['image']
    mask = augmented['mask']

    return img, mask


def normalize(img, mask, weight, img_size) -> tuple:
    img_height, img_width = img_size
    transform = A.Compose([
        A.Resize(img_height, img_width)
    ])

    normalized = transform(image=img, mask=mask)

    img = normalized['image']
    mask = normalized['mask']

    img, mask, weight = convert_to_tensor(img, mask, weight)

    img = img.permute(2, 0, 1).float()  # Garants [CxHxW] and float

    mask = mask.float() / 255.0  # Binary mask
    mask = mask.unsqueeze(0)

    return img, mask, weight


def convert_to_tensor(img,
                      mask,
                      weight
                      ) -> tuple:

    img = torch.from_numpy(img)
    mask = torch.from_numpy(mask)
    weight = torch.tensor(weight, dtype=torch.float)

    return img, mask, weight
