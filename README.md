# SignSense: Sign Language Alphabet Detection

A real-time ASL fingerspelling recognizer that runs on a normal webcam. It tracks the hand with MediaPipe, classifies the letter with a small Keras neural network, and builds up text from the letters you sign. The text can then be read out loud.

I built this as a deep learning course project.

## What it does

- Recognizes 24 static ASL letters (A-Z except J and Z) from a live webcam feed
- Smooths predictions over a few frames so the output doesn't flicker
- Adds a letter only after you hold the sign steady, so passing hand shapes don't get typed
- Builds words and sentences from the committed letters
- Speaks the sentence using text-to-speech

## How it works

Instead of classifying raw images, the project works on hand landmarks:

1. OpenCV reads a frame from the webcam.
2. MediaPipe detects the hand and returns 21 landmark points.
3. The landmarks are turned into a feature vector and passed to an MLP classifier built with Keras.
4. Predictions are smoothed, and a letter is committed once it stays stable long enough.
5. Committed letters form a sentence, which can be spoken aloud.

Working on landmarks instead of pixels keeps the model small and makes it far less sensitive to background and lighting.

## Tech stack

Python, MediaPipe, TensorFlow/Keras, OpenCV, NumPy

## Project structure

```
SignSense/
├── data/               collected landmark data
├── models/             trained model
├── results/            evaluation outputs
├── collect_data.py     record landmark samples for each letter
├── train.py            train the classifier
├── live_predict.py     run live recognition from the webcam
├── utils.py            shared helper functions
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/shreya-js/SignSense.git
cd SignSense
python -m venv venv
venv\Scripts\activate        # on macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

You'll need a webcam.

## Usage

Collect your own data (optional if you use the data already in `data/`):
```bash
python collect_data.py
```

Train the model:
```bash
python train.py
```

Run the live demo:
```bash
python live_predict.py
```

## Results

The classifier reaches about 99.8% accuracy on the held-out test split. Evaluation outputs are in the `results/` folder.

This number comes from my own dataset, so it will probably be lower for other people, other cameras and other lighting. The dataset is small and I collected it myself, so treat it as a result for this setup rather than a general benchmark.

## Limitations

- J and Z are left out because they involve motion, and this model only looks at a single frame.
- It handles fingerspelling only, not full ASL signs or grammar.
- It tracks one hand at a time.

## Possible improvements

- Support J and Z with a sequence model such as an LSTM
- Collect data from more people and in more conditions
- Recognize common words, not just letters

## Author

Shreya
[LinkedIn](https://linkedin.com/in/shreya-shubadev-438502377) | [GitHub](https://github.com/shreya-js)