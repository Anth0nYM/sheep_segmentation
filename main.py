import cv2
import segmentation

if __name__ == '__main__':
    dataloader = segmentation.CustomDataLoader(
        batch_size=3,
        img_size=512,
        subset_size=50,
        shuffle=True,
        augment=True,
        show_progress=True
    )

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()
    test_dataloader = dataloader.get_test_dataloader()

    print(len(train_dataloader))
    print(len(val_dataloader))
    print(len(test_dataloader))

    EPOCHS = 1

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('Press any key to continue, "q" to quit')
        for image_batch, mask_batch in train_dataloader:
            image = image_batch[0].detach().cpu().numpy().transpose(1, 2, 0)
            mask = mask_batch[0].detach().cpu().numpy()

            cv2.imshow('Image', image)
            cv2.imshow('Mask', mask)

            key = cv2.waitKey(0)

            if cv2.waitKey(1) == ord('q'):
                break
        break

    cv2.destroyAllWindows()
