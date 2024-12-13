import torch
import torch.optim as optim
import numpy as np
import segmentation
import segmentation_models_pytorch.losses as losses  # type: ignore
from tqdm import tqdm

if __name__ == '__main__':
    DEVICE = torch.device(device='cuda') if torch.cuda.is_available() else \
        torch.device(device='cpu')
    BATCH_SIZE = 16
    MODEL_NAME = 'unet'

    dataloader = segmentation.CustomDataLoader(
        batch_size=BATCH_SIZE,
        img_size=(512, 512),  # Unet use 512x512 images
        shuffle=True,
        subset_size=30,
        augment=False,
    )

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()
    test_dataloader = dataloader.get_test_dataloader()

    model = segmentation.Model(model_name=MODEL_NAME).to(device=DEVICE)
    metrics = segmentation.MetricSegmentation()
    es = segmentation.EarlyStoppingMonitor(patience=5)

    criterion_1 = losses.JaccardLoss(mode='binary', from_logits=False)
    criterion_2 = losses.DiceLoss(mode='binary', from_logits=False)

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    mode='min',
                                                    patience=5,
                                                    cooldown=5)

    log = segmentation.Log(batch_size=BATCH_SIZE, comment=MODEL_NAME)
    log.log_model(model, next(iter(train_dataloader))[0].to(DEVICE))
    log.log_data_augmentation(augment=dataloader.is_augmented)

    epoch = 0

    while epoch < 20:
        epoch += 1
        model.train()
        train_run_loss = []
        train_metrics: dict[str, list[float]] = {
            metric: [] for metric in metrics.metric_functions.keys()
        }
        train_samples = tqdm(train_dataloader)

        for image, mask in train_samples:
            image, mask = image.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            output = model(image)
            loss_1 = criterion_1(output, mask)
            loss_2 = criterion_2(output, mask)
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()

            batch_metrics = metrics.run_metrics(yt=mask, yp=output)
            for name, value in batch_metrics.items():
                train_metrics[name].append(value)

            train_run_loss.append(loss.item())

            desc = (f'Epoch: {epoch} '
                    f'Train Loss: {np.mean(train_run_loss):.4f} ')
            train_samples.set_description(desc=desc)

        log.log_scalar_train(scalar=np.mean(train_run_loss),
                             epoch=epoch,
                             scalar_name='Loss')

        for name, values in train_metrics.items():
            log.log_scalar_train(scalar=np.mean(values),
                                 epoch=epoch,
                                 scalar_name=name)

        log.log_tensors_train(image=image,
                              mask=mask,
                              output=output,
                              epoch=epoch)

        with torch.no_grad():
            model.eval()
            val_run_loss = []
            val_metrics: dict[str, list[float]] = {
                metric: [] for metric in metrics.metric_functions.keys()
            }
            val_samples = tqdm(val_dataloader)
            for image, mask in val_samples:
                image, mask = image.to(DEVICE), mask.to(DEVICE)
                output = model(image)
                eval_loss = criterion_1(output, mask) + \
                    criterion_2(output, mask)

                batch_metrics = metrics.run_metrics(yt=mask, yp=output)
                for name, value in batch_metrics.items():
                    val_metrics[name].append(value)

                val_run_loss.append(eval_loss.item())
                desc = (f'Epoch: {epoch} '
                        f'Val Loss: {np.mean(val_run_loss):.4f} ')
                val_samples.set_description(desc=desc)

            log.log_scalar_val(scalar=np.mean(val_run_loss),
                               epoch=epoch,
                               scalar_name='Loss')

            for name, values in val_metrics.items():
                log.log_scalar_val(scalar=np.mean(values),
                                   epoch=epoch,
                                   scalar_name=name)

            log.log_tensors_val(image, mask, output, epoch)

            lr_sched.step(np.mean(val_run_loss))
            print(lr_sched.get_last_lr())

            wait = es(np.mean(val_run_loss))
            log.log_scalar_hiper(scalar=wait,
                                 epoch=epoch,
                                 scalar_name="EarlyStoppingWait")

            es(np.mean(val_run_loss))
            if es.must_stop():
                print('Early stopped')
                break

    with torch.no_grad():
        model.eval()
        test_run_loss = []
        test_metrics: dict[str, list[float]] = {
            metric: [] for metric in metrics.metric_functions.keys()
        }
        test_samples = tqdm(test_dataloader, desc="Testing")
        for image, mask in test_samples:
            image, mask = image.to(DEVICE), mask.to(DEVICE)
            output = model(image)

            test_loss = criterion_1(output, mask) + criterion_2(output, mask)
            test_run_loss.append(test_loss.item())

            batch_metrics = metrics.run_metrics(yt=mask, yp=output)
            for name, value in batch_metrics.items():
                test_metrics[name].append(value)

            test_samples.set_description(
                f"Test Loss: {np.mean(test_run_loss):.4f}"
            )

        log.log_scalar_val(scalar=np.mean(test_run_loss),
                           epoch=epoch,
                           scalar_name='Loss/Test')

        for name, values in test_metrics.items():
            log.log_scalar_val(scalar=np.mean(values),
                               epoch=epoch,
                               scalar_name=f"Metric/{name}/Test")

        log.log_tensors_val(image=image, mask=mask, output=output, epoch=epoch)

    log.close()
