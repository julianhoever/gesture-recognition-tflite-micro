import csv
from pathlib import Path


class History:
    def __init__(self) -> None:
        self._metrics: dict[str, dict[str, list[float | str]]] = dict(
            train=dict(), test=dict()
        )

    @property
    def train(self) -> dict[str, list[float | str]]:
        return self._metrics["train"]

    @property
    def test(self) -> dict[str, list[float | str]]:
        return self._metrics["test"]

    def log(
        self, metric: str, train_value: float | str, test_value: float | str
    ) -> None:
        if metric not in self.train:
            self._metrics["train"][metric] = []
            self._metrics["test"][metric] = []

        self._metrics["train"][metric].append(train_value)
        self._metrics["test"][metric].append(test_value)

    def save_csv(self, filename: Path) -> None:
        columns = list(self.train.keys())

        with filename.open("w", newline="") as out_file:
            csv_writer = csv.writer(out_file, delimiter=";")
            csv_writer.writerow(["split", *columns])

            for split in ["train", "test"]:
                value_pairs = zip(*[self._metrics[split][name] for name in columns])
                csv_writer.writerows([split, *values] for values in value_pairs)
