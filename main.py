import torch
import src
import torch.optim as optim
import torch.nn as nn
import segmentation_models_pytorch.losses as losses
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import GroupKFold

if __name__ == '__main__':

    DEVICE = torch.device(device='cuda')\
        if torch.cuda.is_available()\
        else torch.device(device='cpu')
    print(f"Using {DEVICE} as device")
    # models:
    # 'unet',
    # 'unetplusplus',
    # 'fnp',
    # 'pspnet',
    # 'deeplabv3',
    # 'deeplabv3plus',
    # 'linknet',
    # 'manet',
    # 'pan'

    PATH = "dataset/will"
    K = 5
    BATCH_SIZE = 8  # for 512 batch = 16 for 256 batch = 32
    IMAGE_SIZE = 512, 512

    MODEL_NAME = 'fnp'
    EPOCH_LIMIT = 100
    AUGMENT = False
    LAMBDA_SEG = 1.0
    LAMBDA_REG = 1.0
    SEED = 0

    torch.manual_seed(seed=SEED)
    np.random.seed(seed=SEED)
    if DEVICE == torch.device(device='cuda'):
        torch.cuda.manual_seed_all(seed=SEED)

    logger = src.Log(
        k_folds=K,
        comment=f"{MODEL_NAME}{IMAGE_SIZE[0]}_aug{AUGMENT}"
    )

    dataset = src.SheepsDataset(path=PATH,
                                to_augment=False,
                                image_size=IMAGE_SIZE)

    dataloader = src.SheepsLoader(path=PATH,
                                  batch_size=BATCH_SIZE,
                                  img_size=IMAGE_SIZE,
                                  augment=AUGMENT)

    gkf = GroupKFold(n_splits=K, shuffle=True, random_state=SEED)

    for fold, (train_idxs, val_idxs)\
        in enumerate(gkf.split(range(len(dataset)),
                               groups=dataset.get_ids())):

        print(f"Fold {fold + 1}/{K}")

        train_loader = dataloader.get_loader(idxs=train_idxs, split="train")
        val_loader = dataloader.get_loader(idxs=val_idxs, split="val")

        model = src.Model(model_name=MODEL_NAME).to(device=DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        segmentation_report = src.MetricsReport("segmentation")
        regression_report = src.MetricsReport("regression")
        es = src.EarlyStoppingMonitor()
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        mode='min',
                                                        patience=5,
                                                        cooldown=5)

        criterion_seg = losses.DiceLoss(mode='binary', from_logits=False)
        criterion_reg = nn.MSELoss()

        if fold == 0:
            logger.log_text("Data Augmentation is ON"
                            if AUGMENT else "Data Augmentation is OFF")

            logger.log_text(f"Image Size:{IMAGE_SIZE}")

        for epoch in range(EPOCH_LIMIT):
            print(f"Epoch {epoch + 1}")
            model.train()
            train_loss = 0.0
            train_seg_metrics: dict = {
                metric: []
                for metric in segmentation_report.metric_functions.keys()
            }

            train_reg_metrics: dict = {
                metric: []
                for metric in regression_report.metric_functions.keys()
            }

            train = tqdm(train_loader)
            for data in train:
                animal_ids, images, masks, weights = data
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                weights = weights.to(DEVICE)

                # Forward pass
                outputs_seg, outputs_reg = model(images)

                # Loss
                loss_seg = criterion_seg(outputs_seg, masks)
                loss_reg = criterion_reg(outputs_reg.squeeze(), weights)
                loss = LAMBDA_SEG * loss_seg + LAMBDA_REG * loss_reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Metrics
                seg_metrics = segmentation_report.run_metrics(
                    yt=masks,
                    yp=outputs_seg)

                for name, value in seg_metrics.items():
                    train_seg_metrics[name].append(value)

                reg_metrics = regression_report.run_metrics(
                    yt=weights,
                    yp=outputs_reg.squeeze(-1))

                for name, value in reg_metrics.items():
                    train_reg_metrics[name].append(value)

            avg_train_loss = train_loss / len(train)
            avg_train_seg_metrics = {k: sum(v) / len(v)
                                     for k, v in train_seg_metrics.items()}

            avg_train_reg_metrics = {k: sum(v) / len(v)
                                     for k, v in train_reg_metrics.items()}

            logger.log_metrics(fold,
                               phase="train",
                               epoch=epoch,
                               loss=avg_train_loss,
                               metrics={**avg_train_seg_metrics,
                                        **avg_train_reg_metrics})

            logger.log_tensor(fold=fold,
                              phase="train",
                              epoch=epoch,
                              animal_ids=animal_ids,
                              images=images,
                              true_masks=masks,
                              pred_masks=outputs_seg,
                              true_weights=weights,
                              pred_weights=outputs_reg)

            print(f"Training Loss: {avg_train_loss:.4f}")
            desc = (f"Training Fold {fold+1}, Epoch {epoch+1}, "
                    f"Loss {avg_train_loss:.4f}")

            train.set_description(desc=desc)

            with torch.no_grad():
                model.eval()
                val_loss = 0.0

                val_seg_metrics: dict = {
                    metric: []
                    for metric in segmentation_report.metric_functions.keys()
                    }

                val_reg_metrics: dict = {
                    metric: []
                    for metric in regression_report.metric_functions.keys()
                    }

                val = tqdm(val_loader)
                for data in val:
                    animal_ids, images, masks, weights = data
                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)
                    weights = weights.to(DEVICE)

                    outputs_seg, outputs_reg = model(images)

                    loss_seg = criterion_seg(outputs_seg, masks)
                    loss_reg = criterion_reg(outputs_reg.squeeze(), weights)
                    loss = LAMBDA_SEG * loss_seg + LAMBDA_REG * loss_reg
                    val_loss += loss.item()

                    seg_metrics = segmentation_report.run_metrics(
                        yt=masks,
                        yp=outputs_seg)

                    for name, value in seg_metrics.items():
                        val_seg_metrics[name].append(value)

                    reg_metrics = regression_report.run_metrics(
                        yt=weights,
                        yp=outputs_reg.squeeze(-1)
                        )

                    for name, value in reg_metrics.items():
                        val_reg_metrics[name].append(value)

                avg_val_loss = val_loss / len(val)
                avg_val_seg_metrics = {k: sum(v) / len(v)
                                       for k, v in val_seg_metrics.items()}

                avg_val_reg_metrics = {k: sum(v) / len(v)
                                       for k, v in val_reg_metrics.items()}

                logger.log_metrics(fold=fold,
                                   phase="val",
                                   epoch=epoch,
                                   loss=avg_val_loss,
                                   metrics={**avg_val_seg_metrics,
                                            **avg_val_reg_metrics})

                logger.log_tensor(fold=fold,
                                  phase="val",
                                  epoch=epoch,
                                  animal_ids=animal_ids,
                                  images=images,
                                  true_masks=masks,
                                  pred_masks=outputs_seg,
                                  true_weights=weights,
                                  pred_weights=outputs_reg,
                                  n_epochs=5)

                print(f"Validation Loss: {avg_val_loss:.4f}")
                desc = (
                        f"validing Fold {fold+1},"
                        f"Epoch {epoch+1},"
                        f"Loss {avg_val_loss:.4f}")

                val.set_description(desc=desc)

            lr_sched.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            es(avg_val_loss)
            print(f"Current wait: {es.wait}")
            if es.must_stop():
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    logger.close()
