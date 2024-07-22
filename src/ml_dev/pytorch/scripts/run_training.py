from functools import partial

import torch
import matplotlib.pyplot as plt

from ml_dev.pytorch.training import History, train_model
from ml_dev.pytorch.gesture_dataset import GestureDataset
from ml_dev.pytorch.gesture_cnn_model import GestureCnnModel
from ml_dev.pytorch.preprocessing import preprocess
from ml_dev.pytorch.persistence import save_weights
from ml_dev.environment import (
    DATA_ROOT,
    PT_OUTPUTS_DIR,
    PT_MODEL_WEIGHTS_FILE,
    SAMPLE_SHAPE,
    PT_DEVICE,
)


def main() -> None:
    PT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    ds_train, ds_test = _get_datasets()

    model = GestureCnnModel()
    print("Model Params:", _count_parameters(model))

    history = train_model(
        model=model,
        ds_train=ds_train,
        ds_test=ds_test,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=torch.nn.CrossEntropyLoss(),
        batch_size=64,
        epochs=20,
        load_best=True,
        device=PT_DEVICE,
    )

    save_weights(model, PT_MODEL_WEIGHTS_FILE)
    _plot_train_history(history)


def _get_datasets() -> tuple[GestureDataset, GestureDataset]:
    gesture_ds = partial(
        GestureDataset,
        data_root=DATA_ROOT,
        window_size=SAMPLE_SHAPE[0],
        stride=1,
        transform_samples=preprocess,
    )
    return gesture_ds(training=True), gesture_ds(training=False)


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _plot_train_history(history: History) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(history.train["epoch"], history.train["loss"])
    ax.plot(history.test["epoch"], history.test["loss"])
    fig.savefig(PT_OUTPUTS_DIR / "train_history.png")


if __name__ == "__main__":
    main()
