import pandas as pd
import re
import numpy as np

df = pd.read_csv("twitter,csv.csv", encoding='latin-1', on_bad_lines='skip')
print("First few rows:")
print(df.head())
print("Columns:", df.columns)

df = df.drop(columns=["id", "date", "flag", "username"])
print("Missing values per column:")
print(df.isnull().sum())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df['clean_text'] = df['comment'].apply(clean_text)
label_map = {0: 0, 2: 1, 4: 2}
df['target'] = df['target'].map(label_map)
texts = df['clean_text'].tolist()
labels = df['target'].tolist()

def build_vocab(texts):
    vocab = {}
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1
    return vocab

def texts_to_sequences(texts, vocab):
    sequences = []
    for text in texts:
        seq = [vocab.get(word, 0) for word in text.split()]
        sequences.append(seq)
    return sequences

def pad_sequences(sequences, max_len):
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        trunc = seq[:max_len]
        padded[i, :len(trunc)] = np.array(trunc)
    return padded

vocab = build_vocab(texts)
sequences = texts_to_sequences(texts, vocab)
max_length = 100
X = pad_sequences(sequences, max_length)
Y = np.array(labels)

def train_test_split(X, Y, test_ratio=0.2):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_point = int(len(X) * (1 - test_ratio))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_ratio=0.2)

vocab_size = len(vocab) + 1
embedding_dim = 50
hidden_dim = 64
output_dim = 3
learning_rate = 0.005

def init_weights(shape):
    return np.random.randn(*shape) * 0.01

E = init_weights((vocab_size, embedding_dim))
W_x = init_weights((hidden_dim, embedding_dim))
W_h = init_weights((hidden_dim, hidden_dim))
b_h = np.zeros((hidden_dim,))
W_out = init_weights((output_dim, hidden_dim))
b_out = np.zeros((output_dim,))

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - np.tanh(x)**2

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def forward(x):
    h_prev = np.zeros((hidden_dim,))
    hidden_states = []
    for idx in x:
        x_t = E[idx] if idx != 0 else np.zeros((embedding_dim,))
        h_t = tanh(np.dot(W_x, x_t) + np.dot(W_h, h_prev) + b_h)
        hidden_states.append(h_t)
        h_prev = h_t
    logits = np.dot(W_out, h_prev) + b_out
    probs = softmax(logits)
    return probs, hidden_states, h_prev

def cross_entropy_loss(probs, target):
    return -np.log(probs[target] + 1e-10)

num_epochs = 2

for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(len(X_train)):
        x = X_train[i]
        target = y_train[i]
        probs, hidden_states, _ = forward(x)
        loss = cross_entropy_loss(probs, target)
        total_loss += loss
        d_logits = probs.copy()
        d_logits[target] -= 1
        grad_W_out = np.outer(d_logits, hidden_states[-1])
        grad_b_out = d_logits
        d_h = np.dot(W_out.T, d_logits)
        if len(hidden_states) > 1:
            prev_hidden = hidden_states[-2]
            x_last = E[x[-1]] if x[-1] != 0 else np.zeros((embedding_dim,))
            pre_activation = np.dot(W_x, x_last) + np.dot(W_h, prev_hidden) + b_h
        else:
            x_last = E[x[-1]] if x[-1] != 0 else np.zeros((embedding_dim,))
            pre_activation = np.dot(W_x, x_last) + b_h
        d_pre = d_h * dtanh(pre_activation)
        grad_W_x = np.outer(d_pre, x_last)
        if len(hidden_states) > 1:
            grad_W_h = np.outer(d_pre, prev_hidden)
        else:
            grad_W_h = np.zeros_like(W_h)
        grad_b_h = d_pre
        W_out -= learning_rate * grad_W_out
        b_out -= learning_rate * grad_b_out
        W_x   -= learning_rate * grad_W_x
        W_h   -= learning_rate * grad_W_h
        b_h   -= learning_rate * grad_b_h
        if x[-1] != 0:
            E[x[-1]] -= learning_rate * np.dot(W_x.T, d_pre)
    avg_loss = total_loss / len(X_train)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def predict(x):
    probs, _, _ = forward(x)
    return np.argmax(probs)

correct = 0
for i in range(len(X_test)):
    x = X_test[i]
    if predict(x) == y_test[i]:
        correct += 1
test_accuracy = correct / len(X_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

def predict_sentiment(sentence):
    sentence = clean_text(sentence)
    seq = [vocab.get(word, 0) for word in sentence.split()]
    padded = pad_sequences([seq], max_length)
    pred_class = predict(padded[0])
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_map[pred_class]

new_tweets = ["I absolutely love this!", "I'm not really enjoying the movie."]
for tweet in new_tweets:
    sentiment = predict_sentiment(tweet)
    print(f"Tweet: {tweet}\nPredicted Sentiment: {sentiment}\n")
