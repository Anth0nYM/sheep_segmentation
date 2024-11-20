import torch.nn as nn
import torch.optim as optim
import numpy as np
import segmentation
from tqdm import tqdm

if __name__ == '__main__':
    BATCH_SIZE = 2
    dataloader = segmentation.CustomDataLoader(
        batch_size=8,
        img_size=512,  # Unet use 512x512 images
        subset_size=None,
        shuffle=True,
        augment=False,
    )

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()
    test_dataloader = dataloader.get_test_dataloader()
    model = segmentation.Model('unet')
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    EPOCHS = 1

    for epoch in range(EPOCHS):
        model.train()
        running_loss: list = []
        train_samples = tqdm(train_dataloader)
        for image, mask in train_samples:
            # image = image.byte()
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            description = f'Epoch: {epoch} Loss: {np.mean(running_loss):.4f}'
            train_samples.set_description(description)

        model.eval()
        val_samples = tqdm(val_dataloader)
        for image, mask in val_samples:
            pass
