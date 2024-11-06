import cv2
import loader

if __name__ == '__main__':
    dataloader = loader.Dataloader(batch_size=3,
                                   size=512,
                                   shuffle=True,
                                   description=False)

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()
    test_dataloader = dataloader.get_test_dataloader()
    print(len(train_dataloader))
    print(len(val_dataloader))
    print(len(test_dataloader))
    EPOCHS = 0
    for epoch in range(EPOCHS):
        for image, mask in train_dataloader:
            image = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
            mask = mask.detach().cpu().numpy()[0]
            cv2.imshow('image', image)
            cv2.imshow('mask', mask)
            if cv2.waitKey(1) == ord('q'):
                break
