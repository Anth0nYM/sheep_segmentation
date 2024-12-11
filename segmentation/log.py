from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid  # type: ignore
import git
import datetime
import numpy as np
import torch
from typing import Optional, Union


class Git:
    def __init__(self,
                 repot_path: str
                 ) -> None:

        self._repo = git.Repo(repot_path)
        self._sha = self._repo.head.object.hexsha[:8]
        self._date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._commit = self._repo.head.commit

    def timestamp(self) -> str:
        return self._date

    def get_hex(self) -> str:
        return self._sha

    def get_details(self) -> str:
        author = str(self._commit.author)
        commit_message = self._commit.message
        if isinstance(commit_message, bytes):
            commit_message = commit_message.decode()

        return (f'Author: {author},'
                f'Date: {self._commit.committed_datetime},'
                f'Message: {commit_message}')


class Log:
    def __init__(self,
                 batch_size: int,
                 comment: str = '',
                 path: str = 'runs/'
                 ) -> None:

        self._git = Git(repot_path='.')
        self._batch_size = batch_size
        log_dir = (f'{path}{self._git.get_hex()}_{self._git.timestamp()}_'
                   f'{comment}')

        comment = f'{self._git.get_details()}'
        filename_suffix = f'{self._git.timestamp()}'
        self.writer = SummaryWriter(log_dir=log_dir,
                                    comment=comment,
                                    filename_suffix=filename_suffix)

        self._model_saved = False

    def _log_scalar(self,
                    scalar: np.floating,
                    epoch: int,
                    path: str,
                    mean: bool = True
                    ) -> None:

        self.writer.add_scalar(path,
                               np.mean(scalar) if mean else scalar, epoch)
        self.writer.flush()

    def log_scalar_train(self,
                         scalar: np.floating,
                         epoch: int,
                         scalar_name: str,
                         mean: bool = True
                         ) -> None:

        self._log_scalar(scalar=scalar,
                         epoch=epoch,
                         path=f'{scalar_name}/Train',
                         mean=mean)

    def log_scalar_val(self,
                       scalar: np.floating,
                       epoch: int,
                       scalar_name: str,
                       mean: bool = True
                       ) -> None:

        self._log_scalar(scalar=scalar,
                         epoch=epoch,
                         path=f'{scalar_name}/Val',
                         mean=mean)

    def log_scalar_hiper(self,
                         scalar: np.floating,
                         epoch: int,
                         scalar_name: str
                         ) -> None:

        self._log_scalar(scalar=scalar,
                         epoch=epoch,
                         path=f'HIPER/{scalar_name}',
                         mean=False)

    def log_images(self,
                   images: torch.Tensor,
                   epoch: int,
                   path: Optional[str] = None
                   ) -> None:

        img_grid = make_grid(images, nrow=self._batch_size)
        self.writer.add_image(path, img_grid, global_step=epoch)

    def rescale_tensors(self,
                        image: torch.Tensor,
                        mask: torch.Tensor,
                        output: torch.Tensor,
                        epoch: int,
                        split: str
                        ) -> None:
        device = image.device
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=device).view(1, 3, 1, 1)

        std = torch.tensor([0.229, 0.224, 0.225],
                           device=device).view(1, 3, 1, 1)

        image = (image * std + mean).clamp(0, 1)
        image = (image * 255).to(torch.uint8)

        mask = (mask * 255).to(torch.uint8).to(device)
        mask = mask.repeat(1, 3, 1, 1)

        output = (output * 255).to(torch.uint8).to(device)
        output = output.repeat(1, 3, 1, 1)
        images = torch.concat([image, mask, output], dim=0)

        self.log_images(images, epoch, path=f'tensors/{split}')

    def log_tensors_train(self,
                          image: torch.Tensor,
                          mask: torch.Tensor,
                          output: torch.Tensor,
                          epoch: int
                          ) -> None:

        self.rescale_tensors(image, mask, output, epoch, 'train')

    def log_tensors_val(self,
                        image: torch.Tensor,
                        mask: torch.Tensor,
                        output: torch.Tensor,
                        epoch: int
                        ) -> None:

        self.rescale_tensors(image, mask, output, epoch, 'val')

    def close(self) -> None:
        self.writer.close()

    def log_model(self,
                  model: torch.nn.Module,
                  images_input: torch.Tensor,
                  forced_log: bool = False
                  ) -> None:

        if not self._model_saved or forced_log:
            print('Log Model')
            self.writer.add_graph(model, images_input)
            self._model_saved = True

    def log_embedding(self,
                      features: torch.Tensor,
                      class_labels: Union[list[str], torch.Tensor],
                      labels: torch.Tensor
                      ) -> None:

        self.writer.add_embedding(features,
                                  metadata=class_labels,
                                  label_img=labels)

    def log_data_augmentation(self, augment: bool) -> None:
        augment_status = "enabled" if augment else "disabled"
        self.writer.add_text("Data Augmentation",
                             f"Data augmentation is {augment_status}")
        self.writer.flush()
