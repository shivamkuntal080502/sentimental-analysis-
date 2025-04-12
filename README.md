Simple RNN Sentiment Analysis from Scratch
This project demonstrates how to build a simple Recurrent Neural Network (RNN) for tweet sentiment analysis using Python, NumPy, and Pandas. The model is implemented entirely from scratch—without relying on deep learning frameworks—to show the fundamental steps of preprocessing, tokenization, embedding, RNN forward and backward passes, and gradient descent training.

Overview
The code performs the following tasks:

Data Loading & Preprocessing:

Loads a CSV file containing tweets and associated metadata.

Drops unnecessary columns and checks for missing values.

Cleans tweet text by removing mentions, URLs, punctuation, and converting text to lowercase.

Maps sentiment labels (0 for negative, 2 for neutral, and 4 for positive) into new labels (0, 1, and 2 respectively).

Vocabulary Building, Tokenization, and Padding:

Creates a vocabulary that maps each unique word to an integer.

Converts the cleaned text into sequences of integers.

Pads or truncates sequences to a fixed length for uniform input size.

RNN Model Implementation:

Initializes the embedding matrix, RNN weights (input-to-hidden and hidden-to-hidden), and the output layer weights.

Defines a forward pass that processes one tweet at a time.

Computes the cross-entropy loss for sentiment classification.

Training and Evaluation:

Trains the model using a simple gradient descent algorithm with backpropagation through the final time step (a simplified Backpropagation Through Time approach).

Evaluates the model on a test set by computing the accuracy.

Includes an inference function to predict the sentiment of new tweets.

Requirements
Ensure you have Python 3 installed. The necessary packages are:

Pandas

NumPy

re (built-in)

You can install Pandas and NumPy via pip if you haven't already:

bash
Copy
Edit
pip install pandas numpy
Dataset
The dataset is expected to be in a CSV file named twitter,csv.csv with the following assumptions:

The file is encoded with latin-1.

It contains columns named such as id, date, flag, username, comment, and target.

Unnecessary columns (e.g., id, date, flag, and username) are dropped from the analysis.

Make sure that your CSV file is correctly formatted and saved in the same directory as the code.

Running the Code
Clone the Repository or Download the Files:
Place the provided Python script and the CSV file in the same directory.

Run the Script:
Execute the script using Python:

bash
Copy
Edit
python your_script_name.py
The script will display the first few rows of the data, the status of missing values, the training progress (including average loss per epoch), the test accuracy, and predicted sentiment for sample tweets.

Code Structure
Data Loading and Preprocessing:
The code loads the CSV file, cleans the tweet text using a regular expression, and converts sentiment labels.

Vocabulary Construction and Tokenization:
Functions handle converting tweets to sequences of integer indices using a generated vocabulary and pad sequences to a fixed length.

RNN Model Implementation:
The simple RNN includes:

An embedding layer.

An RNN cell performing a forward pass over the input sequence.

A final linear output layer producing classification logits.

A simplified backpropagation through time (BPTT) that computes gradients based on the last hidden state.

Training Loop:
The model trains for a predefined number of epochs using stochastic gradient descent, updating weights after processing each sample.

Evaluation and Inference:
After training, the model evaluates performance on a test set and provides functions to predict the sentiment of new tweets.

Customization
Hyperparameters:
You can adjust parameters such as embedding_dim, hidden_dim, output_dim, learning_rate, and max_length according to your dataset and performance needs.

Epochs:
The number of training epochs (num_epochs) can be increased for better training but may require longer run times.

Limitations
The RNN implementation only backpropagates through the final time step. This simplified approach is mainly for educational purposes.

The model might not generalize well on a larger dataset without improvements such as full Backpropagation Through Time (BPTT), regularization, or more advanced architectures (e.g., LSTM or GRU).

Conclusion
This project offers a foundational understanding of how sequence models like RNNs work for sentiment analysis tasks. It provides a step-by-step illustration from data loading and preprocessing to training a simple RNN and evaluating its performance.

