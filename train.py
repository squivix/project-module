import itertools
import json
import math
import os
import time

import numpy
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.mlp import MLPBinaryClassifier, weight_reset
from utils import calc_binary_classification_metrics, undersample_dataset


def kfold_grid_search(dataset, device, in_features, checkpoint_file_path=None, k=5, max_epochs=20, batch_size=32,
                      hidden_layer_combs=None,
                      unit_combs=None,
                      dropout_combs=None,
                      threshold_combs=None,
                      learning_rate_combs=None,
                      weight_decay_combs=None,
                      focal_alpha_combs=None,
                      focal_gamma_combs=None,
                      ):
    if hidden_layer_combs is None:
        hidden_layer_combs = [1]
    if unit_combs is None:
        unit_combs = [2048]
    if dropout_combs is None:
        dropout_combs = [0.2]
    if threshold_combs is None:
        threshold_combs = [0.5]
    if learning_rate_combs is None:
        learning_rate_combs = [0.001]
    if weight_decay_combs is None:
        weight_decay_combs = [0.0]
    if focal_alpha_combs is None:
        focal_alpha_combs = [0.5]
    if focal_gamma_combs is None:
        focal_gamma_combs = [2.0]
    grid_search_start_time = int(time.time() * 1000)
    checkpoint_dir = f"checkpoints/grid-search/{grid_search_start_time}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    param_to_metrics = {}
    if checkpoint_file_path is not None:
        with open(checkpoint_file_path, 'rb') as checkpoint_file:
            param_to_metrics = json.load(checkpoint_file)
    max_iters = math.prod([len(c) for c in [hidden_layer_combs, unit_combs, dropout_combs, threshold_combs, learning_rate_combs, weight_decay_combs, focal_alpha_combs, focal_gamma_combs]])
    i = 0
    for hidden_layers, units, dropout, threshold, focal_alpha, focal_gamma in itertools.product(hidden_layer_combs, unit_combs, dropout_combs,
                                                                                                threshold_combs, focal_alpha_combs, focal_gamma_combs):
        model_builder = MLPBinaryClassifier(in_features=in_features, hidden_layers=hidden_layers, units_per_layer=units,
                                            dropout=dropout, threshold=threshold, focal_alpha=focal_alpha, focal_gamma=focal_gamma)
        for learning_rate, weight_decay in itertools.product(learning_rate_combs, weight_decay_combs):
            param_key = f"(hidden_layers={hidden_layers}, units={units}, dropout={dropout}, threshold={threshold}, learning_rate={learning_rate}, weight_decay={weight_decay}, focal_alpha={focal_alpha}, focal_gamma={focal_gamma})"
            print(f"({i}/{max_iters}) {param_key}")
            if not param_key in param_to_metrics:
                eval_metrics = kfold_train_eval(model_builder, dataset,
                                                batch_size=batch_size, device=device, k=k, max_epochs=max_epochs,
                                                learning_rate=learning_rate, weight_decay=weight_decay, )
                param_to_metrics[param_key] = eval_metrics
            else:
                eval_metrics = param_to_metrics[param_key]
            print(eval_metrics)

            with open(f"{checkpoint_dir}/{i}.json", "w") as f:
                json.dump(param_to_metrics, f)

            if eval_metrics["test_mcc"] >= 0.5:
                print(f"Over 0.5 mcc found with {param_key}")
                with open(f"{checkpoint_dir}/good_mcc_{i}.json", "w") as f:
                    json.dump(param_to_metrics, f)
            i += 1
    best_params = max(param_to_metrics, key=lambda k: param_to_metrics[k]['test_mcc'])
    print()
    print(f"Best params: {best_params}")
    print(f"Best performance: {param_to_metrics[best_params]}")


def kfold_train_eval(model, dataset, device, k=5, learning_rate=0.001, weight_decay=0.0, max_epochs=20,
                     batch_size=32):
    metrics = ["loss", "accuracy", "precision", "recall", "f1", "mcc"]
    test_metrics = {m: [] for m in metrics}
    kfold = StratifiedKFold(n_splits=k, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, dataset.labels)):
        train_dataset = Subset(dataset, train_ids)
        train_dataset.labels = np.array(dataset.labels)[train_ids]
        train_loader = DataLoader(undersample_dataset(train_dataset), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_ids), batch_size=batch_size, shuffle=True)

        model.apply(weight_reset)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in tqdm(range(max_epochs), desc=f"Fold {fold + 1}"):
            model.train()
            batches = iter(train_loader)

            for i, (batch_x, batch_y) in enumerate(batches):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model.forward(batch_x)
                loss = model.loss_function(logits, batch_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            test_batches = iter(test_loader)

            batch_test_metrics = {m: np.empty(len(test_batches)) for m in metrics}
            for i, (x_test, y_test) in enumerate(tqdm(test_batches, desc=f"Testing")):
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                test_logits = model.forward(x_test)
                test_loss = model.loss_function(test_logits, y_test)
                test_preds = model.predict(test_logits)
                accuracy, precision, recall, f1, mcc = calc_binary_classification_metrics(y_test, test_preds)

                batch_test_metrics["loss"][i] = test_loss
                batch_test_metrics["accuracy"][i] = accuracy
                batch_test_metrics["precision"][i] = precision
                batch_test_metrics["recall"][i] = recall
                batch_test_metrics["f1"][i] = f1
                batch_test_metrics["mcc"][i] = mcc
            for m in metrics:
                test_metrics[m].append(batch_test_metrics[m].mean())
    return {**{f"test_{m}": numpy.mean(test_metrics[m]) for m in metrics}, }


def train_classifier(model, train_loader, test_loader, device, learning_rate=0.001, weight_decay=0,
                     max_epochs=1000,
                     checkpoint_every=None, eval_every=1):
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    metrics = ["loss", "accuracy", "precision", "recall", "f1", "mcc"]
    train_metrics = {m: [] for m in [*metrics, "epoch"]}
    test_metrics = {m: [] for m in [*metrics, "epoch"]}

    training_start_time = int(time.time() * 1000)
    checkpoint_dir = f"checkpoints/{model.__class__.__name__}/{training_start_time}"

    if checkpoint_every is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # with  open(f"{checkpoint_dir}/train_dataset.json", 'w') as temp_file:
        #     json.dump(train_loader.dataset.to_dict(), temp_file)
        # with  open(f"{checkpoint_dir}/test_dataset.json", 'w') as temp_file:
        #     json.dump(test_loader.dataset.to_dict(), temp_file)

    for epoch in range(max_epochs):
        if checkpoint_every is not None and epoch % checkpoint_every == 0:
            torch.save(model, f"{checkpoint_dir}/{epoch}.pickle")
        batches = iter(train_loader)

        batch_train_metrics = {m: np.empty(len(batches)) for m in metrics}
        for i, (batch_x, batch_y) in enumerate(tqdm(batches, desc=f"Epoch {epoch + 1:,} training")):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model.forward(batch_x)
            loss = model.loss_function(logits, batch_y)
            preds = model.predict(logits)
            accuracy, precision, recall, f1, mcc = calc_binary_classification_metrics(batch_y, preds)

            batch_train_metrics["loss"][i] = loss
            batch_train_metrics["accuracy"][i] = accuracy
            batch_train_metrics["precision"][i] = precision
            batch_train_metrics["recall"][i] = recall
            batch_train_metrics["f1"][i] = f1
            batch_train_metrics["mcc"][i] = mcc
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for m in metrics:
            train_metrics[m].append(batch_train_metrics[m].mean())
        train_metrics["epoch"].append(epoch)

        print(f"Train: {epoch + 1:,}/{max_epochs:,}: loss:{train_metrics["loss"][-1]}")
        if eval_every is not None and epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                test_batches = iter(test_loader)

                batch_test_metrics = {m: np.empty(len(test_batches)) for m in metrics}
                for i, (x_test, y_test) in enumerate(tqdm(test_batches, desc=f"Epoch {epoch + 1:,} testing")):
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    test_logits = model.forward(x_test)
                    test_loss = model.loss_function(test_logits, y_test)
                    test_preds = model.predict(test_logits)
                    test_accuracy, test_precision, test_recall, test_f1, test_mcc = calc_binary_classification_metrics(
                        y_test,
                        test_preds)

                    batch_test_metrics["loss"][i] = test_loss
                    batch_test_metrics["accuracy"][i] = test_accuracy
                    batch_test_metrics["precision"][i] = test_precision
                    batch_test_metrics["recall"][i] = test_recall
                    batch_test_metrics["f1"][i] = test_f1
                    batch_test_metrics["mcc"][i] = test_mcc
                for m in metrics:
                    test_metrics[m].append(batch_test_metrics[m].mean())
                test_metrics["epoch"].append(epoch)
                print(
                    f"Test: {epoch + 1:,}/{max_epochs:,}: {", ".join([f"{k}:{v[-1]}" for k, v in test_metrics.items()])}")
            model.train()

    if checkpoint_every is not None:
        torch.save(model, f"{checkpoint_dir}/final.pickle")
    return model, {
        **{f"train_{m}": train_metrics[m] for m in metrics},
        **{f"test_{m}": test_metrics[m] for m in metrics},

        "train_epoch": train_metrics["epoch"],
        "test_epoch": test_metrics["epoch"],
    }
