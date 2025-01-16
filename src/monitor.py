from typing import Optional


class EarlyStoppingMonitor:
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-4
                 ) -> None:

        self.__patience = patience
        self.__min_delta = min_delta
        self.wait = 0
        self.__best_loss: Optional[float] = None
        self.__stop_training = False

    def __call__(self, loss: float) -> None:
        if self.__best_loss is None:
            self.__best_loss = loss
        elif loss > self.__best_loss - self.__min_delta:
            self.wait += 1
            if self.wait >= self.__patience:
                self.__stop_training = True
        else:
            self.__best_loss = loss
            self.wait = 0

    def must_stop(self) -> bool:
        return self.__stop_training
