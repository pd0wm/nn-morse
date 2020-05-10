#!/usr/bin/env python3
import random
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data


from morse import ALPHABET, generate_sample
from itertools import groupby

num_tags = len(ALPHABET)

# 0: blank label
tag_to_idx = {c: i + 1 for i, c in enumerate(ALPHABET)}
idx_to_tag = {i + 1: c for i, c in enumerate(ALPHABET)}


def prediction_to_str(seq):
    if not isinstance(seq, list):
        seq = seq.tolist()

    # remove duplicates
    seq = [i[0] for i in groupby(seq)]

    # remove blanks
    seq = [s for s in seq if s != 0]

    # convert to string
    seq = "".join(idx_to_tag[c] for c in seq)

    return seq


def get_training_sample(*args, **kwargs):
    _, spec, y = generate_sample(*args, **kwargs)

    spec = torch.from_numpy(spec)
    spec = spec.permute(1, 0)

    y_tags = [tag_to_idx[c] for c in y]
    y_tags = torch.tensor(y_tags)

    return spec, y_tags


class Net(nn.Module):
    def __init__(self, num_tags, spectrogram_size):
        super(Net, self).__init__()

        num_tags = num_tags + 1  # 0: blank
        hidden_dim = 256
        lstm_dim1 = 256

        self.dense1 = nn.Linear(spectrogram_size, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.dense4 = nn.Linear(hidden_dim, lstm_dim1)
        self.lstm1 = nn.LSTM(lstm_dim1, lstm_dim1, batch_first=True)
        self.dense5 = nn.Linear(lstm_dim1, num_tags)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))

        x, _ = self.lstm1(x)

        x = self.dense5(x)
        x = F.log_softmax(x, dim=2)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Dataset(data.Dataset):
    def __len__(self):
        return 2048

    def __getitem__(self, index):
        length = random.randrange(10, 20)
        pitch = random.randrange(100, 950)
        wpm = random.randrange(10, 40)
        noise_power = random.randrange(0, 200)
        amplitude = random.randrange(10, 150)
        return get_training_sample(length, pitch, wpm, noise_power, amplitude)


def collate_fn_pad(batch):
    xs, ys = zip(*batch)

    input_lengths = torch.tensor([t.shape[0] for t in xs])
    output_lengths = torch.tensor([t.shape[0] for t in ys])

    seqs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True)

    return input_lengths, output_lengths, seqs, ys


if __name__ == "__main__":
    batch_size = 64
    spectrogram_size = generate_sample()[1].shape[0]

    device = torch.device("cuda")
    writer = SummaryWriter()

    # Set up trainer & evaluator
    model = Net(num_tags, spectrogram_size).to(device)
    print("Number of params", model.count_parameters())

    # Lower learning rate to 1e-4 after about 1500 epochs
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ctc_loss = nn.CTCLoss()

    train_loader = torch.utils.data.DataLoader(
        Dataset(),
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_fn_pad,
    )

    random.seed(0)

    epoch = 0

    # Resume training
    if epoch != 0:
        model.load_state_dict(torch.load(f"models/{epoch:06}.pt", map_location=device))

    model.train()
    while True:
        for (input_lengths, output_lengths, x, y) in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)

            m = torch.argmax(y_pred[0], 1)
            y_pred = y_pred.permute(1, 0, 2)

            loss = ctc_loss(y_pred, y, input_lengths, output_lengths)

            loss.backward()
            optimizer.step()

        writer.add_scalar("training/loss", loss.item(), epoch)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"models/{epoch:06}.pt")

        print(prediction_to_str(y[0]))
        print(prediction_to_str(m))
        print(loss.item())
        print()
        epoch += 1
