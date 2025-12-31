# RNN / LSTM for Kids: Movie Review Feelings (IMDB Sentiment)

This notebook teaches how an **RNN** (Recurrent Neural Network) learns from **sequences** (ordered things),
like words in a sentence.

We will use the built-in Keras dataset **IMDB**:
- movie reviews (text)
- labels: **0 = negative** 😞, **1 = positive** 😄

Even though it’s kid-friendly, it uses real deep learning ideas:
- **tokenization** (words → numbers)
- **padding** (make sequences the same length)
- **Embedding** (numbers → meaning vectors)
- **RNN / LSTM** (reads words in order)
- **Dropout** (prevents memorizing)
- training **history**, graphs, evaluation
- choose a test review and **predict** its sentiment

---

## Big intuition

A CNN is great for images (space patterns).

An RNN is great for sequences (time/order patterns).

Think of an RNN like a kid reading a sentence **word by word**:
- It keeps a small **memory** of what it read so far.
- That memory changes as each new word arrives.
## 0) Install / Imports

If you are using Colab, TensorFlow is usually installed.

If running locally (terminal):

```bash
pip install tensorflow matplotlib numpy ipywidgets
```
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
print("TensorFlow version:", tf.__version__)
## 1) Load the IMDB dataset (built-in)

Keras gives IMDB reviews already converted into **integers** (word IDs).
That is perfect for neural networks.

We choose:
- `num_words = 10000` → keep only the 10,000 most common words.

Each review becomes something like:
`[1, 14, 20, 2, 56, ...]` (a list of integers)

Label:
- 0 = negative
- 1 = positive
num_words = 10000  # vocabulary size (how many different word IDs we keep)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)

print("Number of training reviews:", len(x_train))
print("Number of test reviews    :", len(x_test))

print("Example review (as word IDs):", x_train[0][:30], "...")
print("Example label:", y_train[0])

# Reviews are variable length (not all same size)
lengths = [len(r) for r in x_train]
print("Min length:", min(lengths), "Max length:", max(lengths), "Average length:", sum(lengths)/len(lengths))
### What is `shape` for sequences?

A single review is not a rectangle like an image.
It is a **list** of word IDs, and different reviews have different lengths.

But neural networks love rectangles (same length).

So we use **padding** to make every review the same length.
## 2) Padding: make all reviews the same length

We pick a maximum length `max_len`.

- If a review is shorter → add zeros at the front (or end)
- If a review is longer → cut it

`0` will mean: “empty padding”.

After padding, `x_train_padded` becomes a big rectangle:
`(num_reviews, max_len)`
max_len = 200  # keep last 200 words (good for a demo)

x_train_padded = tf.keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen=max_len, padding="pre", truncating="pre"
)
x_test_padded = tf.keras.preprocessing.sequence.pad_sequences(
    x_test, maxlen=max_len, padding="pre", truncating="pre"
)

print("x_train_padded shape:", x_train_padded.shape)  # (25000, 200)
print("x_test_padded shape :", x_test_padded.shape)   # (25000, 200)

print("First padded review (first 30 IDs):", x_train_padded[0][:30])
## 3) Embedding: turning word IDs into meaning vectors

Right now each word is just an integer ID:
- `42` doesn't "mean" anything by itself.

**Embedding** is like giving each word a little “meaning arrow” in space:
- Each word becomes a vector like `[0.1, -0.3, 0.7, ...]`

Embedding layer shape idea:
- Input: `(batch_size, max_len)`  → integers
- Output: `(batch_size, max_len, embed_dim)` → vectors

Example:
- `max_len = 200`
- `embed_dim = 32`

Then each review becomes a **row of 200 word-vectors**.
## 4) RNN / LSTM: reading in order

An RNN reads the sequence step-by-step.

Think of a little robot reading words:
- It sees one word vector
- Updates its memory
- Moves to the next word

A basic RNN can forget long-distance info.
So we often use **LSTM** (Long Short-Term Memory),
which has a smarter memory system.

We'll use **LSTM** because it works well and is common.

### Output choices
- `return_sequences=False` (default): LSTM returns only the final memory state.
That final state is like: “My final understanding of the whole review.”
## 5) Build the model (Keras)

We will build:

1. **Embedding** (word IDs → vectors)
2. **LSTM** (reads 200 steps)
3. **Dropout** (anti-memorizing)
4. **Dense(1)** with sigmoid (probability of positive)

Why sigmoid?
- It outputs a number between 0 and 1
- close to 1 → positive
- close to 0 → negative
embed_dim = 32

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_len,)),

    # Turn word IDs into vectors
    tf.keras.layers.Embedding(input_dim=num_words, output_dim=embed_dim),

    # LSTM reads the sequence
    tf.keras.layers.LSTM(64),

    # Dropout to reduce overfitting
    tf.keras.layers.Dropout(0.3),

    # One output: probability of positive review
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
## 6) Train the model

We train for a few epochs (demo-friendly).

We also use `validation_split=0.2`:
- 80% training
- 20% validation (mini test while training)
history = model.fit(
    x_train_padded, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)
## 7) Plot training history

- Loss should go down
- Accuracy should go up

If validation accuracy goes down while training accuracy goes up,
the model might be memorizing (overfitting).
hist = history.history

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(hist["loss"], label="train loss")
plt.plot(hist["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(hist["accuracy"], label="train acc")
plt.plot(hist["val_accuracy"], label="val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")

plt.show()
## 8) Evaluate on test set

Now we check accuracy on reviews the model never trained on.
test_loss, test_acc = model.evaluate(x_test_padded, y_test, verbose=0)
print("Test accuracy:", float(test_acc))
print("Test loss    :", float(test_loss))
## 9) Decode reviews back into words (so humans can read them)

IMDB dataset has a word index dictionary.

We will:
- get `word_index` mapping: word → id
- invert it to: id → word
- decode a review (list of IDs) into text

Important detail:
Keras IMDB uses special reserved IDs:
- 0: padding
- 1: start of sequence
- 2: unknown word
- 3: unused

So when decoding we often subtract 3 from IDs, or use an offset.
Keras uses an offset of 3.
# Get mapping word -> id
word_index = tf.keras.datasets.imdb.get_word_index()

# Invert to id -> word
id_to_word = {idx + 3: word for word, idx in word_index.items()}
id_to_word[0] = "<PAD>"
id_to_word[1] = "<START>"
id_to_word[2] = "<UNK>"
id_to_word[3] = "<UNUSED>"

def decode_review(review_ids):
    # Convert list of ints to a readable sentence
    words = [id_to_word.get(i, "<UNK>") for i in review_ids]
    return " ".join(words)

# Show one decoded training review
example_idx = 0
print("Label:", y_train[example_idx], "(1=positive, 0=negative)")
print(decode_review(x_train[example_idx][:60]), "...")
## 10) Predict one test review

We will:
- take a test review
- pad it
- model outputs a probability `p`
- if `p >= 0.5` → positive else negative
idx = 0
review_ids = x_test[idx]
true_label = int(y_test[idx])

review_padded = tf.keras.preprocessing.sequence.pad_sequences(
    [review_ids], maxlen=max_len, padding="pre", truncating="pre"
)

p = float(model.predict(review_padded, verbose=0)[0][0])
pred_label = 1 if p >= 0.5 else 0

print("True label:", true_label, "->", "positive" if true_label==1 else "negative")
print("Pred prob positive:", round(p, 4))
print("Pred label:", pred_label, "->", "positive" if pred_label==1 else "negative")

print("\nReview snippet (decoded):")
print(decode_review(review_ids[:80]), "...")
## 11) Choose another test review and predict (interactive)

If `ipywidgets` works:
- use a slider to pick any test index
- see decoded text + prediction

If widgets do not work:
- change `idx = ...` manually and run again.
def show_prediction(idx: int, words_to_show: int = 120):
    review_ids = x_test[idx]
    true_label = int(y_test[idx])

    review_padded = tf.keras.preprocessing.sequence.pad_sequences(
        [review_ids], maxlen=max_len, padding="pre", truncating="pre"
    )

    p = float(model.predict(review_padded, verbose=0)[0][0])
    pred_label = 1 if p >= 0.5 else 0

    print("="*80)
    print(f"Test index: {idx}")
    print("True:", "positive" if true_label==1 else "negative")
    print("Pred prob positive:", round(p, 4))
    print("Pred:", "positive" if pred_label==1 else "negative")
    print("-"*80)
    print(decode_review(review_ids[:words_to_show]))
    print("="*80)

try:
    import ipywidgets as widgets
    from IPython.display import display

    slider = widgets.IntSlider(value=0, min=0, max=len(x_test)-1, step=1, description="Test idx:")
    words = widgets.IntSlider(value=120, min=30, max=300, step=10, description="Words:")
    ui = widgets.interactive_output(show_prediction, {"idx": slider, "words_to_show": words})
    display(slider, words, ui)

except Exception as e:
    print("ipywidgets not available here. Manual mode works!")
    idx = 123
    show_prediction(idx)
## 12) (Optional) Swap LSTM for SimpleRNN or GRU

Try changing the model:

- `tf.keras.layers.SimpleRNN(64)` (simpler, can forget more)
- `tf.keras.layers.GRU(64)` (like LSTM, often faster)

The rest stays the same.

---

# Quick quiz (check understanding)

1. Why do we need **padding** for reviews?
2. What is the difference between a **word ID** and an **embedding vector**?
3. What does an **LSTM** do that helps with sequences?
4. Why do we use **sigmoid** at the end instead of softmax?
5. If the model outputs `p = 0.92`, what does that mean?
6. If `max_len = 200`, what is the shape of `x_train_padded`?
