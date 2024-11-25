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
        subset_size=16,
        shuffle=True,
        augment=False,
    )

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()
    test_dataloader = dataloader.get_test_dataloader()

    model = segmentation.Model('unet')
    metrics = segmentation.MetricSegmentation()
    es = segmentation.EarlyStoppingMonitor(patience=5)

    criterion_1 = nn.L1Loss()
    criterion_2 = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    mode='min',
                                                    patience=5)

    EPOCHS = 2

    for epoch in range(EPOCHS):
        model.train()
        train_run_loss = []
        train_run_iou = []
        train_run_dice = []
        model.train()
        train_samples = tqdm(train_dataloader)
        for image, mask in train_samples:
            optimizer.zero_grad()
            output = model(image)
            loss_1 = criterion_1(output, mask)
            loss_2 = criterion_2(output, mask)
            iou, dice = metrics.run_metrics(yt=mask,
                                            yp=output,
                                            epoch=epoch,
                                            split='train')

            train_run_iou.append(iou)
            train_run_dice.append(dice)
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            train_run_loss.append(loss.item())
            desc = (f'Epoch: {epoch} '
                    f'Loss: {np.mean(train_run_loss):.4f} '
                    f'IOU: {np.mean(train_run_iou):.4f} '
                    f'Dice: {np.mean(train_run_dice):.4f}')

            train_samples.set_description(desc=desc)

        model.eval()
        val_samples = tqdm(val_dataloader)
        for image, mask in val_samples:
            pass
