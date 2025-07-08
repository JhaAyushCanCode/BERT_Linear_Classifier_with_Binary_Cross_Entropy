# GoEmotions Multi-Label Emotion Classifier

This project contains two different deep learning models for detecting **multiple emotions in a single sentence** using the **GoEmotions dataset**. One version uses **BERT** and the other uses **RoBERTa** for better accuracy. Both are fine-tuned for the same task but differ slightly in architecture and performance.

## What this project does

* Takes Reddit comments and predicts which emotions are present
* Supports **multi-label classification** (a sentence can have more than one emotion)
* Built using PyTorch and HuggingFace Transformers
* Works best on a **GPU**

## Models Included

### 1. BERT Version

* Uses `bert-base-uncased`
* Has a dropout + linear layer on top of BERT
* Trained for 4 epochs
* Gives decent results with good training time

### 2. RoBERTa Version (Improved Accuracy)

* Uses `roberta-base`
* Similar structure to BERT version but replaces BERT with RoBERTa
* Trained for 8 epochs
* Generally gives **better accuracy** (especially on Macro-F1)
* Slightly slower to train

## What files are included

### Code Files

* `bert_classifier.py` – full code for BERT-based classifier
* `roberta_classifier.py` – full code for RoBERTa-based classifier

### Output Files (After Running)

Each version generates the following:

* `goemotions_bert_model.pt` or `goemotions_roberta_model.pt` – final saved model
* `checkpoint_epoch_*.pt` – checkpoints saved at every epoch
* `f1_scores.csv` – CSV file logging F1 scores and loss per epoch
* `f1_plot.png` – line plot of F1 score trends

## How it works

* Loads the GoEmotions dataset (via HuggingFace)
* Splits into 90 percent training and 10 percent testing
* Tokenizes text using the appropriate tokenizer (`bert-base-uncased` or `roberta-base`)
* Encodes the labels as a binary vector (multi-hot format)
* Feeds input through BERT or RoBERTa
* Uses a sigmoid layer on top to output 28 scores (one per emotion)
* Applies a threshold (0.5) to convert scores into final binary predictions

## Training settings

| Setting       | BERT Version      | RoBERTa Version   |
| ------------- | ----------------- | ----------------- |
| Max Length    | 128 tokens        | 128 tokens        |
| Batch Size    | 16                | 16                |
| Epochs        | 4                 | 8                 |
| Learning Rate | 2e-5              | 2e-5              |
| Optimizer     | AdamW             | AdamW             |
| Loss Function | BCEWithLogitsLoss | BCEWithLogitsLoss |

## Evaluation metrics

Each model is evaluated using:

* **Micro-F1**: Looks at overall label-wise performance
* **Macro-F1**: Treats all emotion labels equally, even if some are rare
* **Training Loss**: BCEWithLogitsLoss is used for multi-label problems

### Example console output

```
Epoch 1/8
Training Loss: 0.1293
Micro-F1: 0.5437, Macro-F1: 0.2208
...
```

## Visualization

Both versions output a plot (`f1_plot.png`) showing how Micro-F1 and Macro-F1 scores change with each epoch.

## Dataset

* Source: [GoEmotions Dataset](https://huggingface.co/datasets/go_emotions)
* Collected by Google
* 58,000+ Reddit comments
* Each comment has 1–3 emotion labels
* There are 28 emotion categories in total

## Requirements

Install the needed Python packages with:

```bash
pip install torch transformers datasets scikit-learn matplotlib tqdm pandas
```

## How to run

To train the BERT version:

```bash
python bert_classifier.py
```

To train the RoBERTa version:

```bash
python roberta_classifier.py
```

Both scripts will save the model weights, evaluation logs, and F1 score plots.

## Notes

* Both models run better on GPU. On CPU, training will be very slow.
* RoBERTa usually performs slightly better but takes more time to train.
* You can experiment by increasing epochs or adjusting dropout for better results.

## License

This project is for learning and research use. If you use this work in academic projects, please cite the GoEmotions dataset and HuggingFace Transformers library.
