import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc, f1_score, matthews_corrcoef, accuracy_score
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryCalibrationError
from tqdm import tqdm


def expected_calibration_error(y_true, y_probs, n_bins=10):
    """Computes Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:  # Avoid division by zero
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def test_classifier(model, test_dataset, device, batch_size=256):
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    model.eval()
    metrics = ["loss", "accuracy", "precision", "recall", "f1", "mcc", "ece", "pr_auc"]
    test_metrics = {m: float("NaN") for m in metrics}

    # Store all true labels and predicted probabilities
    all_y_true = []
    all_y_probs = []

    with torch.no_grad():
        test_batches = iter(test_loader)
        batch_test_metrics = {m: np.empty(len(test_batches)) for m in metrics}

        for i, (x_test, y_test) in enumerate(tqdm(test_batches, desc=f"Testing")):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            test_logits = model.forward(x_test)
            test_loss = model.loss_function(test_logits, y_test)

            # Store true labels and model scores for PR curve
            all_y_true.append(y_test.cpu())
            all_y_probs.append(test_logits.cpu())

            batch_test_metrics["loss"][i] = test_loss
        all_y_true = torch.concatenate(all_y_true)
        all_y_probs = torch.concatenate(all_y_probs)

    for m in metrics:
        if m in batch_test_metrics:
            test_metrics[m] = batch_test_metrics[m].mean()

    precision, recall, thresholds = precision_recall_curve(all_y_true, all_y_probs)
    pr_auc = auc(recall, precision)

    # Find optimal threshold based on F1-score
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)  # Avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5  # Handle edge case

    # Convert probabilities to binary using the optimal threshold
    all_y_preds = (all_y_probs >= optimal_threshold).int()

    # Calculate other metrics
    precision_score = precision[optimal_idx]
    recall_score = recall[optimal_idx]
    f1 = f1_score(all_y_true, all_y_preds)
    mcc = matthews_corrcoef(all_y_true, all_y_preds)
    accuracy = accuracy_score(all_y_true, all_y_preds)
    ece = BinaryCalibrationError(norm='l1')(all_y_probs.squeeze(), all_y_true.squeeze()).item()

    test_metrics["accuracy"] = accuracy
    test_metrics["precision"] = precision_score
    test_metrics["recall"] = recall_score
    test_metrics["f1"] = f1
    test_metrics["mcc"] = mcc
    test_metrics["ece"] = ece
    test_metrics["pr_auc"] = pr_auc

    return test_metrics
