import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam

from utils import calc_binary_classification_metrics
from tqdm import tqdm

def mlp_train(model, train_loader, test_loader, device, learning_rate=0.001, max_epochs=1000):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    metrics = ["accuracy", "loss", "precision", "recall", "f1"]
    train_metrics = {m: [] for m in metrics}
    test_metrics = {m: [] for m in metrics}

    for epoch in tqdm(range(max_epochs)):
        # print(f"{epoch}/{max_epochs}")
        batches = iter(train_loader)

        batch_train_metrics = {m: np.empty(len(batches)) for m in metrics}
        for i, (batch_x, batch_y) in enumerate(batches):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model.forward(batch_x)
            loss = model.loss_function(logits, batch_y)
            preds = model.predict(logits)
            accuracy, precision, recall, f1 = calc_binary_classification_metrics(batch_y, preds)

            batch_train_metrics["loss"][i] = loss
            batch_train_metrics["accuracy"][i] = accuracy
            batch_train_metrics["precision"][i] = precision
            batch_train_metrics["recall"][i] = accuracy
            batch_train_metrics["f1"][i] = f1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for m in metrics:
            train_metrics[m].append(batch_train_metrics[m].mean())

        model.eval()
        with torch.no_grad():
            test_batches = iter(test_loader)
            batch_test_metrics = {m: np.empty(len(batches)) for m in metrics}
            for i, (x_test, y_test) in enumerate(test_batches):
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                test_logits = model.forward(x_test)
                test_loss = model.loss_function(test_logits, y_test)
                test_preds = model.predict(test_logits)
                test_accuracy, test_precision, test_recall, test_f1 = calc_binary_classification_metrics(y_test,
                                                                                                         test_preds)

                batch_test_metrics["loss"][i] = test_loss
                batch_test_metrics["accuracy"][i] = test_accuracy
                batch_test_metrics["precision"][i] = test_precision
                batch_test_metrics["recall"][i] = test_accuracy
                batch_test_metrics["f1"][i] = test_f1
            for m in metrics:
                test_metrics[m].append(batch_test_metrics[m].mean())

        model.train()

    return model, {
        **{f"train_{m}": train_metrics[m] for m in metrics},
        **{f"test_{m}": test_metrics[m] for m in metrics}
    }
