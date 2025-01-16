from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import git
import datetime
import torchvision.transforms as T
from PIL import ImageDraw, ImageFont
import os
import torch
from collections import defaultdict
import pandas as pd


class Git:
    def __init__(self, repo_path: str) -> None:
        self._repo = git.Repo(repo_path)
        self._sha = self._repo.head.object.hexsha[:4]
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

        return (f"Author: {author}, Date: {self._commit.committed_datetime}, "
                f"Message: {commit_message}")


class Log:
    def __init__(self,
                 k_folds: int,
                 comment: str = '',
                 path: str = 'logs/',
                 ) -> None:

        self.__git = Git(repo_path='.')
        self.__k_folds = k_folds
        self.__fold_results: dict =\
            defaultdict(lambda: {"train": {}, "val": {}})

        self.__log_dir = (
            f"{path}{self.__git.get_hex()}_{comment}_{self.__git.timestamp()}"
        )

        self.__writers = {
            f"fold_{i + 1}": SummaryWriter(
                os.path.join(self.__log_dir, f"fold_{i + 1}")
            )
            for i in range(self.__k_folds)
        }

        os.makedirs(self.__log_dir, exist_ok=True)

    def log_text(self, text: str) -> None:
        writer = self.__writers["fold_1"]
        writer.add_text("Experiment Details", text)

    def log_metrics(self,
                    fold: int,
                    phase: str,
                    epoch: int,
                    loss: float,
                    metrics: dict
                    ) -> None:

        writer = self.__writers[f"fold_{fold + 1}"]
        writer.add_scalar(f"{phase}/loss", loss, epoch)
        for metric_name, value in metrics.items():
            writer.add_scalar(f"{phase}/{metric_name}", value, epoch)

        self.__fold_results[fold][phase] = {
            **self.__fold_results[fold][phase],
            "loss": loss,
            **metrics
        }

    def log_tensor(self,
                   fold: int,
                   phase: str,
                   epoch: int,
                   animal_ids: torch.Tensor,
                   images: torch.Tensor,
                   true_masks: torch.Tensor,
                   pred_masks: torch.Tensor,
                   true_weights: torch.Tensor,
                   pred_weights: torch.Tensor,
                   n_epochs: int = 5,
                   n_images: int = 5
                   ) -> None:

        if epoch % n_epochs != 0:
            return  # Apenas loga a cada `n` épocas.

        writer = self.__writers[f"fold_{fold + 1}"]

        # Seleciona aleatoriamente `n_images` índices
        max_images = min(images.size(0), n_images)
        random_indices = torch.randperm(images.size(0))[:max_images]

        # Seleciona amostras aleatórias com base nos índices
        images = images[random_indices]
        true_masks = true_masks[random_indices]
        pred_masks = pred_masks[random_indices]
        animal_ids = animal_ids[random_indices]
        true_weights = true_weights[random_indices]
        pred_weights = pred_weights[random_indices]

        # Normaliza tensores para visualização
        true_masks = true_masks.repeat(1, 3, 1, 1)

        pred_masks = pred_masks.repeat(1, 3, 1, 1)

        images = (images - images.min()) / (images.max() - images.min())

        # Combina imagens, máscaras reais e preditas
        concat_tensors = torch.cat([images, true_masks, pred_masks], dim=0)
        grid = make_grid(concat_tensors, nrow=max_images, padding=10)

        # Adiciona texto com ID do animal, peso real e peso predito
        grid_img = T.ToPILImage()(grid)
        draw = ImageDraw.Draw(grid_img)
        font = ImageFont.truetype(font="arialbd.ttf", size=20)

        img_width = grid_img.width // max_images
        img_height = grid_img.height // 3

        for idx in range(max_images):
            x_pos = idx * img_width
            y_pos = img_height - 60
            text_lines = [
                f"ID: {animal_ids[idx].item()}",
                f"True: {true_weights[idx].item():.2f}",
                f"Pred: {pred_weights[idx].item():.2f}"
                ]
            for line_idx, line_text in enumerate(text_lines):
                draw.text((x_pos + 5, y_pos + line_idx * 20),
                          line_text, fill=(0, 0, 255), font=font)

        # Converte de volta para tensor
        grid_tensor = T.ToTensor()(grid_img)

        # Loga no TensorBoard
        writer.add_image(f"{phase}/Epoch_{epoch}_Results", grid_tensor, epoch)

    def summarize_results(self):
        """
        Consolida e salva os resultados de todas as dobras em um DataFrame.

        Returns:
            pd.DataFrame: DataFrame com os resultados consolidados.
        """
        summary_rows = []

        for phase in ["train", "val"]:
            metrics_all_folds = defaultdict(list)

            # Coleta resultados para cada métrica e perda
            for fold in range(self.__k_folds):
                fold_results = self.__fold_results[fold][phase]
                for metric, value in fold_results.items():
                    metrics_all_folds[metric].append(value)

            # Calcula média e desvio padrão para cada métrica
            summary_row = {"phase": phase}
            for metric, values in metrics_all_folds.items():
                mean = sum(values) / len(values)
                std = (
                    sum((x - mean) ** 2 for x in values) / len(values)
                ) ** 0.5
                summary_row[metric] = f"{mean:.4f} ± {std:.4f}"

            summary_rows.append(summary_row)

        # Salva os resultados em CSV
        df_summary = pd.DataFrame(summary_rows)
        csv_path = os.path.join(self.__log_dir, "summary_results.csv")
        df_summary.to_csv(csv_path, index=False)

        print(f"Summary results saved to {csv_path}")
        return df_summary

    def close(self) -> None:
        # Consolida e salva os resultados antes de fechar
        self.summarize_results()
        for writer in self.__writers.values():
            writer.close()
