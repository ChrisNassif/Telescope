#!/usr/bin/env python3
"""Train an LSTM (or logistic-regression) sequence classifier on per-token perplexity data.

Consumes CSVs produced by ``build_dataset.py`` and trains a bidirectional LSTM head
over concatenated datasets, saving the trained model checkpoint at the end.

This replaces the previous ``sequence_modeling_trainer.ipynb`` notebook.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from ablations._common import add_common_args, get_logger, resolve_common_args


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_of_layers, output_dim, bidirectional=False, device="cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.number_of_layers = number_of_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, number_of_layers,
            batch_first=True, device=device, bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim, output_dim, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.sigmoid(self.fc(self.sigmoid(hn[-1])))


class LogisticRegressionModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.linear = nn.LazyLinear(1, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


class TelescopeSequenceModelingDataset(Dataset):
    def __init__(self, dataset_file: Path, device: str, signal_column: str):
        df = pd.read_csv(dataset_file)
        self.sequences = [
            torch.tensor(ast.literal_eval(s), dtype=torch.float32, device=device).view(-1, 1)
            for s in df[signal_column]
        ]
        self.labels = torch.tensor(df["labels"].to_numpy(), dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class TelescopeAverageDataset(Dataset):
    def __init__(self, dataset_file: Path, device: str, signal_column: str):
        df = pd.read_csv(dataset_file)
        self.sequences = torch.tensor(
            [np.average(ast.literal_eval(s)) for s in df[signal_column]],
            dtype=torch.float32, device=device,
        ).view(-1, 1)
        self.labels = torch.tensor(df["labels"].to_numpy(), dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def make_collate_fn(device: str):
    def _collate(batch):
        packed = pack_sequence([b[0] for b in batch], enforce_sorted=False)
        labels = torch.tensor([b[1] for b in batch], device=device)
        return packed, labels
    return _collate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets-dir",
        type=str,
        required=True,
        help="Directory containing <dataset>_<model>_dataset/full.csv folders (from build_dataset.py).",
    )
    p.add_argument(
        "--dataset-names",
        type=str,
        nargs="+",
        default=[
            "hc3_plus_smollm_360M_dataset",
            "hc3_smollm_360M_dataset",
            "ai_human_smollm_360M_dataset",
            "detect_llm_text_smollm_360M_dataset",
            "esl_gpt4o_smollm_360M_dataset",
        ],
        help="Names of dataset folders under --datasets-dir to concatenate.",
    )
    p.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "logistic"],
        default="lstm",
    )
    p.add_argument("--input-dim", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=300)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--bidirectional", action="store_true", default=True)
    p.add_argument("--no-bidirectional", dest="bidirectional", action="store_false")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--lr-gamma", type=float, default=0.98)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument(
        "--signal-column", type=str, default="telescope_perplexity_per_token",
        help="Column in full.csv to use as the input sequence.",
    )
    p.add_argument(
        "--dataset-type", type=str, choices=["sequence", "average"], default="sequence",
        help="'sequence' feeds the full per-token series; 'average' collapses to a scalar.",
    )
    p.add_argument("--model-name", type=str, default="smollm_360M_lstm_extra_features_all_datasets.pt",
                   help="Filename for the saved model checkpoint.")
    add_common_args(p, "sequence_modeling/train_sequence_model")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = resolve_common_args(args, "sequence_modeling/train_sequence_model")
    logger = get_logger("train_sequence_model", args.log_level)

    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 32)

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}")

    datasets_dir = Path(args.datasets_dir)
    dataset_class = TelescopeSequenceModelingDataset if args.dataset_type == "sequence" else TelescopeAverageDataset
    parts = []
    for name in args.dataset_names:
        csv = datasets_dir / name / "full.csv"
        if not csv.exists():
            logger.warning(f"missing: {csv}")
            continue
        parts.append(dataset_class(csv, device, args.signal_column))
    if not parts:
        raise SystemExit("No datasets loaded.")
    full = ConcatDataset(parts)

    train_ds, test_ds = random_split(full, [args.train_split, 1 - args.train_split])
    collate = make_collate_fn(device) if args.dataset_type == "sequence" else None
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    if args.model_type == "lstm":
        model = LSTMModel(
            input_dim=args.input_dim, hidden_dim=args.hidden_dim,
            number_of_layers=args.num_layers, output_dim=1,
            bidirectional=args.bidirectional, device=device,
        )
    else:
        model = LogisticRegressionModel(device=device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)

    model.train()
    for epoch in range(args.epochs):
        for batch_idx, (data, labels) in enumerate(train_dl):
            optimizer.zero_grad()
            out = model(data).view(-1)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        logger.info(f"epoch {epoch+1}/{args.epochs} loss={loss.item():.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, labels in test_dl:
            out = model(data).view(-1).cpu().numpy()
            lbl = labels.cpu().numpy()
            correct += int(np.sum((out > 0.5) == (lbl > 0.5)))
            total += len(lbl)
    if total:
        logger.info(f"test accuracy: {correct / total:.4f} (n={total})")

    ckpt = out_dir / args.model_name
    torch.save(model, ckpt)
    logger.info(f"saved model to {ckpt}")


if __name__ == "__main__":
    main()
