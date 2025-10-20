# 🐦 Twitter Sentiment Analysis

## 📌 Overview
This project performs **sentiment analysis on Twitter data** related to video games and tech products.  
It uses classical machine learning methods (Random Forest) combined with **text preprocessing and TF-IDF vectorization**.

---

---

## 📊 Dataset

- **Source:** `twitter_training.csv`  
- **Initial Shape:** `(74,681, 4)`  
- **Columns:**
  - Unnamed numeric column (removed)
  - `Borderlands` → Game or product name  
  - `Positive` → Sentiment label  
  - Tweet text  

- **Preprocessed subset:** 8,000 samples  
  - 2,000 tweets per sentiment class (`Positive`, `Negative`, `Neutral`, `Irrelevant`)  

---

## 🧹 Data Preprocessing

1. **Drop unnecessary columns** (ID columns).  
2. **Rename columns** for clarity:
   - `Borderlands` → `Game_Name`  
   - `Positive` → `Sentiment`  
   - Tweet text → `tweet`  
3. **Handle missing values** by dropping rows with empty tweets.  
4. **Create a balanced subset** with 2,000 samples per sentiment class.  

---

## 🔢 Categorical to Numerical Transformation

- Game names are **one-hot encoded** to convert categorical variables into numeric features.  
- The encoded columns include games like `Amazon`, `Borderlands`, `Overwatch`, etc.  

---

## ✍️ Text Preprocessing

- Uses **spaCy** for:
  1. **Lemmatization** – reduce words to their base form.  
  2. **Stopword and punctuation removal** – retain meaningful words.  

- Final cleaned text is stored in a column called `final`.

---

## 📈 Feature Engineering

- Apply **TF-IDF vectorization** on the preprocessed `final` column to transform text into numerical features.  
- Combine **TF-IDF features** with **one-hot encoded game features** to create the final input matrix for the model.

---

## 🔀 Train-Test Split

- 80% training and 20% testing split:
  - `x_train`, `y_train`: training features and labels  
  - `x_test`, `y_test`: testing features and labels  

---


## 🧰 Dependencies

Python 3.10+
pandas
scikit-learn
spaCy (en_core_web_sm)

