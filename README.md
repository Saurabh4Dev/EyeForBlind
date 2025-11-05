# Eye for Blind

A deep learning solution for generating image captions with speech output to assist visually impaired individuals in understanding visual content.

## Overview
'''
This project aims to assist visually impaired people by building a deep learning model that can describe images in speech.
Using a CNN-RNN architecture with an attention mechanism, the model generates captions for images and converts them to audio using text-to-speech.
The solution enables blind users to understand image content.
'''

## Features

- **CNN-RNN Architecture**: Uses InceptionV3 for feature extraction and GRU with attention mechanism for caption generation
- **Attention Mechanism**: Bahdanau attention to focus on relevant image regions while generating captions
- **Search Strategies**: Both greedy search and beam search for caption generation
- **Text-to-Speech**: Converts generated captions to speech for accessibility
- **Comprehensive Evaluation**: BLEU score evaluation with detailed metrics
- **Visualization**: Attention map visualization to understand model focus


## Requirements

### System Requirements
- Python 3.8 or higher
- TensorFlow 2.13.0 or higher
- Jupyter notebook
- Google colab and drive

## Quick Start

### 1. Dataset Setup
Download the Flickr8K dataset and organize as follows:
```
flickr8k_data/
├── Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── captions.txt
```

### 2. Configuration
```
Config variables and parameters:
    IMAGES_PATH = "/path/to/your/flickr8k_data/Images"
    CAPTIONS_PATH = "/path/to/your/flickr8k_data/captions.txt"
    ...
```

### 3. Project completion steps
```
The major steps to perform can be briefly summarised in the following steps:

Data Understanding: Load the data and understand the representation.
Data Preprocessing:Process both images and captions to the desired format.
Train-Test Split: Combine both images and captions to create the train and test dataset.
Model Building: Create your image captioning model by building Encoder, Attention and Decoder model.
Model Evaluation: Evaluate the models using greedy search and BLEU score.
```
## Architecture

### Model Components

1. **CNN Encoder**
   - Pre-trained InceptionV3 without top layer
   - Extracts 2048-dimensional feature vectors
   - Projects features to embedding space

2. **Attention Mechanism**
   - Bahdanau attention mechanism
   - Allows decoder to focus on relevant image regions
   - Generates attention weights for visualization

3. **RNN Decoder**
   - GRU-based decoder with attention
   - Generates captions word by word
   - Uses teacher forcing during training

### Training Pipeline

1. **Data Preprocessing**
   - Image resizing and normalization
   - Caption tokenization and padding
   - Feature extraction using CNN

2. **Model Training**
   - Teacher forcing for efficient training
   - Validation loss monitoring
   - Early stopping to prevent overfitting
   - Model checkpointing for best performance

3. **Evaluation**
   - BLEU score calculation (1-gram to 4-gram)
   - Attention visualization
   - Qualitative assessment of generated captions

## Implementation Summary

This project implements an image captioning model with attention and text-to-speech output. The key steps are:

### Data Loading and Preprocessing:

Flickr8k dataset images and captions are loaded.
Captions are preprocessed: tokenized, filtered, and padded.
Images are resized and preprocessed for the InceptionV3 model.
Image features are extracted using a pre-trained InceptionV3 model to save memory and computational time.
Image paths and captions are split into training and testing sets.
A data pipeline using tf.data.Dataset is created to efficiently load image features and captions in batches.

### Model Architecture:

An Encoder (CNN_Custom_Encoder) uses a fully connected layer to process the extracted image features.
An Attention mechanism (BahdanauAttention) calculates the attention weights, allowing the model to focus on relevant parts of the image during caption generation. Different attention mechanisms (Luong, Dot-Product) were also defined.
A Decoder (RNN_Custom_Decoder) uses a GRUCell and an Embedding layer to generate captions word by word, using the context vector from the attention mechanism and the previous word.

### Training and Optimization:

The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss with masking for padded sequences.
Teacher forcing is applied during training to guide the decoder with the true next word.
Training performed for defined number of epochs as it provided better results(predicted captions).
Early stopping is provided as choice for implementation to monitor test loss and prevent overfitting and saving the best model checkpoint but the Predicted captions are more near to ground truth without early stopping so it exists as choice but not used.
Training progress and loss curves are visualized.

### Model Evaluation and Prediction:

A greedy search function is defined to generate captions for new images.
The BLEU score is used to evaluate the quality of the generated captions against the real captions, with different weightings for n-grams.
A function is created to convert the generated captions into audio using gTTS.
Attention maps are visualized to show which parts of the image the model is focusing on for each word in the generated caption.

## Evaluation Metrics

### BLEU Score Calculation
The model uses BLEU scores to evaluate caption quality:
- **BLEU-1**: Measures unigram precision
- **BLEU-2**: Measures bigram precision  
- **BLEU-3**: Measures trigram precision
- **BLEU-4**: Measures 4-gram precision
```python
test_image = predict_caption_audio(len(image_test), True, weights = (0.1, 0.2, 0.3, 0.4))
.....
score = sentence_bleu(reference, candidate, weights=weights) #set your weights
```

### Attention Visualization
The attention mechanism can be visualized to understand which parts of the image the model focuses on when generating each word.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- **Flickr8K Dataset**: University of Illinois
- **TensorFlow Team**: For the deep learning framework
- **Attention Mechanism**: Bahdanau et al. (2014)
- **Image Captioning Research**: Various academic contributions

## Contact

Created by Saurabh Pandey(@Saurabh4Dev) - feel free to contact me!

---

**Note**:  Google colab and Google drive has been used to train and predict the model.
