class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait before stopping when no improvement.
            min_delta (float): Minimum change in the monitored metric to be considered an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        """
        Call this method at the end of each epoch to check for early stopping.

        Args:
            val_loss (float): The validation loss for the current epoch.

        Returns:
            (bool): True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if validation loss improves
        else:
            self.counter += 1  # Increase counter if no improvement

        if self.counter >= self.patience:
            return True  # Stop training

        return False  # Continue training

    def reset(self):
        """Resets the early stopping parameters for a new training session."""
        self.best_loss = float('inf')
        self.counter = 0
