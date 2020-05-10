#!/usr/bin/env python3

import numpy as np
from scipy import signal
import scipy.io.wavfile
import argparse

from morse import SAMPLE_FREQ, get_spectrogram, ALPHABET
from main import prediction_to_str, num_tags, Net
import matplotlib.pyplot as plt

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("input")
    args = parser.parse_args()

    rate, data = scipy.io.wavfile.read(args.input)
    length = len(data) / rate
    new_length = int(length * SAMPLE_FREQ)

    data = signal.resample(data, new_length)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))

    spec = get_spectrogram(data)
    spec_orig = spec.copy()
    spectrogram_size = spec.shape[0]

    device = torch.device("cpu")

    spec = torch.from_numpy(spec)
    spec = spec.permute(1, 0)
    spec = spec.unsqueeze(0)

    model = Net(num_tags, spectrogram_size)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    y_pred = model(spec)
    y_pred_l = np.exp(y_pred[0].tolist())

    # TODO: proper beam search
    m = torch.argmax(y_pred[0], 1)
    print(prediction_to_str(m))

    # Only show letters with > 5% prob somewhere in the sequence
    labels = np.asarray(["<blank>", "<space>"] + list(ALPHABET[1:]))
    sum_prob = np.max(y_pred_l, axis=0)
    show_letters = sum_prob > .05

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.pcolormesh(spec_orig)
    plt.subplot(2, 1, 2)
    plt.plot(y_pred_l[:, show_letters])
    plt.legend(labels[show_letters])
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.show()
