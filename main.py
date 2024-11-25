import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import segmentation
from tqdm import tqdm

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    BATCH_SIZE = 8
    dataloader = segmentation.CustomDataLoader(
        batch_size=BATCH_SIZE,
        img_size=512,  # Unet use 512x512 images
        subset_size=16,
        shuffle=True,
        augment=False,
    )

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()
    test_dataloader = dataloader.get_test_dataloader()

    model = segmentation.Model(model_name='unet').to(device=device)
    metrics = segmentation.MetricSegmentation()
    es = segmentation.EarlyStoppingMonitor(patience=5)

    criterion_1 = nn.L1Loss()
    criterion_2 = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    mode='min',
                                                    patience=5)

    epoch = 0

    while True:
        epoch += 1
        model.train()
        train_run_loss = []
        train_run_iou = []
        train_run_dice = []
        model.train()
        train_samples = tqdm(train_dataloader)

        for image, mask in train_samples:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(image)

            loss_1 = criterion_1(output, mask)
            loss_2 = criterion_2(output, mask)
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()

            iou, dice = metrics.run_metrics(yt=mask, yp=output)

            train_run_iou.append(iou)
            train_run_dice.append(dice)
            train_run_loss.append(loss.item())

            desc = (f'Epoch: {epoch} '
                    f'Train Loss: {np.mean(train_run_loss):.4f} '
                    f'Train IOU: {np.mean(train_run_iou):.4f} '
                    f'Train Dice: {np.mean(train_run_dice):.4f}')

            train_samples.set_description(desc=desc)

        with torch.no_grad():
            model.eval()
            val_run_loss = []
            val_run_iou = []
            val_run_dice = []
            val_samples = tqdm(val_dataloader)
            for image, mask in val_samples:
                image, mask = image.to(device), mask.to(device)
                output = model(image)
                eval_loss = criterion_1(output, mask) + \
                    criterion_2(output, mask)

                iou, dice = metrics.run_metrics(yt=mask, yp=output)
                val_run_iou.append(iou)
                val_run_dice.append(dice)
                val_run_loss.append(eval_loss.item())
                desc = (f'Epoch: {epoch} '
                        f'Val Loss: {np.mean(val_run_loss):.4f} '
                        f'Val IOU: {np.mean(val_run_iou):.4f} '
                        f'Val Dice: {np.mean(val_run_dice):.4f}')

                val_samples.set_description(desc=desc)

            lr_sched.step(np.mean(val_run_loss))
            print(lr_sched.get_last_lr())
            es(np.mean(val_run_loss))
            if es.must_stop():
                print('Early stopping')
                break
