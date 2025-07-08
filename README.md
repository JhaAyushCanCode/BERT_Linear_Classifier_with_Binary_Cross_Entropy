# GoEmotions Multi-Label Emotion Classifier with BERT
This project trains a BERT model to detect multiple emotions in a single sentence using the GoEmotions dataset. It's built using PyTorch and HuggingFace Transformers, and works best if you're using a GPU.

## What this does
It takes Reddit comments (from the GoEmotions dataset) and trains a model to predict which emotions are present.

Each sentence can have more than one emotion (multi-label).

Uses a pretrained BERT model and adds a final classifier layer for 28 emotion types.

## How it works
Loads data from HuggingFace (go_emotions dataset).

Preprocesses the text using the BERT tokenizer.

Trains BERT with a small classification head on top.

Evaluates after each epoch using Micro-F1 and Macro-F1 scores.

Saves:

The trained model.

A CSV log of F1 scores.

A plot of how scores changed over time.

## What's included
### Code
main.py – this has all the code to run the training, testing, evaluation, and saving.

## Output Files (after you run it)
goemotions_bert_model.pt – final model weights

checkpoint_epoch_1.pt, ..._2.pt, etc. – model checkpoints for each epoch

f1_scores.csv – tracks micro and macro F1 scores plus training loss

f1_plot.png – graph of F1 scores across epochs

## What the model looks like
Pretrained bert-base-uncased

A dropout layer (to help generalize)

A linear layer to output 28 emotion scores

Outputs are run through sigmoid to get probabilities for each emotion

Training setup
Max input length: 128 tokens

Batch size: 16

Epochs: 4

Learning rate: 2e-5

Optimizer: AdamW

Loss: BCEWithLogitsLoss (because we have multiple labels per input)

## How it evaluates performance
### We calculate:

Micro-F1: Treats all labels across all samples equally

Macro-F1: Treats all labels equally, regardless of how many times they appear

These are common metrics for multi-label tasks

## What you’ll see during training
### Example console output:

yaml
Copy
Edit
Epoch 1/4
Training Loss: 0.1308
Micro-F1: 0.4873, Macro-F1: 0.1917
...
Visual output
Once training is done, you’ll get this:

f1_plot.png
A line plot showing:

Micro-F1 over epochs

Macro-F1 over epochs
Gives a visual idea of how the model improved (or didn’t) during training.

## Dataset used
GoEmotions from Google

58k+ Reddit comments, each labeled with 1–3 emotions from a list of 28

Dataset is loaded and split 90 percent for training and 10 percent for testing

## How to use this
Requirements
### You’ll need the following Python libraries:


pip install torch transformers datasets scikit-learn matplotlib tqdm pandas
### To run the training

python main.py

### After it finishes, you’ll find:

## The trained model files

F1 score logs in CSV

A plot of F1 over time

## Final notes
The model trains on GPU if available.

It's a solid base model. You can build more advanced stuff on top of it.

The dataset is already cleaned and labeled, so it’s great for experimenting.

#### License
Feel free to use this for research, learning, or experimentation. If you use it in a paper or product, be sure to credit the GoEmotions dataset and HuggingFace.
