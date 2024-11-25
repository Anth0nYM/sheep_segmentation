class EarlyStoppingMonitor:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = None
        self.stopped_epoch = 0
        self.stop_training = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
        else:
            self.best_loss = loss
            self.wait = 0

    def must_stop(self):
        return self.stop_training
