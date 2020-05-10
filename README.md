# Neural Network morse decoder
Use a neural network to decode morse code! The input is preprocessed by taking the FFT in 20 ms windows. These inputs go into a few dense layers followed by an LSTM. The network does not use time based convolutions on the input, so it can be used in a streaming fashion before all the audio is available.


# Train
Run `./main.py`. Open tensorboard to view progess. The network will converge after about 1000 epochs.

# Test
Run `./decode_audio.py --model models/000690.pt audio/hello_world.wav`. This will print the prediction, and show a plot of the spectrogram and tokens as predicted by the model.

![](hello_world.png)
