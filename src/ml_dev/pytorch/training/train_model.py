from typing import Any
from functools import partial
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

from .history import History


def train_model(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_test: Dataset,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    batch_size: int,
    epochs: int,
    load_best: bool = True,
    device: Any = "cpu",
) -> History:
    data_loader = partial(DataLoader, batch_size=batch_size)
    dl_train = data_loader(ds_train, shuffle=True)
    dl_test = data_loader(ds_test, shuffle=False)

    model.to(device)

    history = History()
    best_test_accuracy = 0
    best_model_state = deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        correct_predicted = 0
        num_samples = 0

        for samples, labels in dl_train:
            samples = samples.to(device)
            labels = labels.to(device)

            model.zero_grad()

            predictions = model(samples)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted_labels = torch.argmax(predictions, dim=-1)
            correct_predicted += float(torch.sum(predicted_labels == labels))
            num_samples += len(labels)

        train_loss = running_loss / len(dl_train)
        train_accuracy = correct_predicted / num_samples

        model.eval()

        running_loss = 0.0
        correct_predicted = 0
        num_samples = 0

        with torch.no_grad():
            for samples, labels in dl_test:
                samples = samples.to(device)
                labels = labels.to(device)

                predictions = model(samples)
                loss = loss_fn(predictions, labels)

                running_loss += loss.item()
                predicted_labels = torch.argmax(predictions, dim=-1)
                correct_predicted += float(torch.sum(predicted_labels == labels))
                num_samples += len(labels)

        test_loss = running_loss / len(dl_test)
        test_accuracy = correct_predicted / num_samples

        history.log("epoch", epoch, epoch)
        history.log("loss", train_loss, test_loss)
        history.log("accuracy", train_accuracy, test_accuracy)

        if load_best and test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model_state = deepcopy(model.state_dict())

        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss: {train_loss:.04f} ; train_accuracy: {train_accuracy:.04f} ; "
            f"test_loss: {test_loss:.04f} ; test_accuracy: {test_accuracy:.04f}"
        )

    if load_best:
        model.load_state_dict(best_model_state)

    return history
