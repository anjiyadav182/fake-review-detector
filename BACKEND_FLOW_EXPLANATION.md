# 🔄 Complete Backend Flow: Form Submission → ML Prediction

## 📝 User Action: Form Submit

### Step 1: User Fills Form (`templates/form.html`)

```html
<form action="/predict" method="POST">
  <textarea name="review_text">"This product is amazing! I love it!"</textarea>
  <input type="number" name="overall" value="5" />
  <button type="submit">Submit Review</button>
</form>
```

**What happens:**
- User types review: `"This product is amazing! I love it!"`
- User selects rating: `5` stars
- Clicks "Submit Review" button

**Browser sends:**
```
POST /predict
Content-Type: application/x-www-form-urlencoded

review_text=This product is amazing! I love it!
overall=5
```

---

## 🎯 Step 2: Flask Receives Request (`app.py` line 411-414)

```python
@app.route('/predict', methods=['GET', 'POST'])
def prediction_function():
    prediction = None
    if request.method == 'POST':
        # Extract form data
        review_text = request.form['review_text']  # "This product is amazing! I love it!"
        overall = int(request.form['overall'])      # 5
```

**What happens:**
- Flask route `/predict` receives POST request
- Extracts `review_text` from form
- Extracts `overall` rating from form
- Converts rating to integer

**Current State:**
```python
review_text = "This product is amazing! I love it!"
overall = 5
helpful_ratio = 0.0  # default value
```

---

## 🔢 Step 3: Feature Extraction (Lines 423-435)

### A. Numerical Features Extraction

```python
# Extract features from review text
review_length = len(review_text)                    # 35 characters
word_count = len(review_text.split())               # 6 words
avg_word_length = np.mean([len(word) for word in review_text.split()])  # 4.5
```

**Example Calculation:**
```
review_text = "This product is amazing! I love it!"
review_length = 35  (character count)
word_count = 6      (word count)
avg_word_length = mean([4, 7, 2, 7, 1, 3]) = 4.0
```

### B. Sentiment Analysis (Line 427)

```python
sentiment_score = analyzer.polarity_scores(review_text)['compound']
# VADER Sentiment Analyzer calculates:
# - positive: 0.636
# - neutral: 0.364
# - negative: 0.0
# - compound: 0.636 (positive sentiment)
```

**VADER Analysis:**
- Analyzes text for positive/negative words
- "amazing", "love" → positive words
- Returns compound score: `0.636` (positive)

### C. Sentiment Label Encoding (Lines 429-435)

```python
if sentiment_score >= 0.05:
    sentiment_label = 2      # Positive
elif sentiment_score <= -0.05:
    sentiment_label = 0      # Negative
else:
    sentiment_label = 1      # Neutral

# Our example: 0.636 >= 0.05 → sentiment_label = 2
```

**Current Features:**
```python
review_length = 35
helpful_ratio = 0.0
word_count = 6
avg_word_length = 4.0
sentiment_score = 0.636
sentiment_label = 2
overall = 5
```

---

## 📊 Step 4: Feature Scaling (Lines 437-440)

### Create DataFrame for Numerical Features

```python
numerical_features = pd.DataFrame(
    [[review_length, helpful_ratio, word_count, avg_word_length]], 
    columns=['reviewLength', 'helpfulRatio', 'wordCount', 'avgWordLength']
)
```

**DataFrame:**
```
   reviewLength  helpfulRatio  wordCount  avgWordLength
0           35           0.0          6            4.0
```

### Apply Scaling Pipeline

```python
scaled_features = scaling_pipeline.transform(numerical_features)
```

**What happens:**
- `scaling_pipeline.pkl` contains a **StandardScaler**
- Formula: `(x - mean) / std`
- Normalizes features to similar scales
- Example output:
```
   reviewLength  helpfulRatio  wordCount  avgWordLength
0         -0.5          0.0        0.2           -0.1
```

**Why Scaling?**
- Different features have different ranges
- Review length: 0-1000
- Word count: 0-200
- Scaling makes all features comparable

---

## 🧹 Step 5: Text Preprocessing (Line 443)

### Function: `preprocess_and_lemmatize()` (Lines 115-124)

```python
cleaned_review_text = preprocess_and_lemmatize(review_text)
```

**Step-by-step preprocessing:**

#### Step 5.1: Lowercase Conversion
```python
text = text.lower()
# "This product is amazing! I love it!" 
# → "this product is amazing! i love it!"
```

#### Step 5.2: Expand Contractions (Lines 103-112)
```python
text = expand_contractions(text)
# "i love it!" → "i love it!" (no contractions here)
# Example: "can't" → "cannot", "won't" → "will not"
```

#### Step 5.3: Remove Punctuation & Numbers
```python
text = re.sub(r'[^a-zA-Z\s]', '', text)
# "this product is amazing! i love it!"
# → "this product is amazing i love it"
```

#### Step 5.4: Tokenization
```python
tokens = nltk.word_tokenize(text)
# → ['this', 'product', 'is', 'amazing', 'i', 'love', 'it']
```

#### Step 5.5: Remove Stopwords
```python
tokens = [token for token in tokens if token not in stop_words]
# Stopwords: ['this', 'is', 'i', 'it'] removed
# → ['product', 'amazing', 'love']
```

#### Step 5.6: Lemmatization (spaCy)
```python
doc = nlp(' '.join(tokens))
tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
# "amazing" → "amaze"
# "love" → "love"
# → "product amaze love"
```

**Final Cleaned Text:**
```
Original: "This product is amazing! I love it!"
Cleaned:  "product amaze love"
```

---

## 🔤 Step 6: Text Vectorization (Line 446)

### TF-IDF Vectorization

```python
vectorized_text = vectorization_pipeline.transform([cleaned_review_text]).toarray()
```

**What happens:**
- `vectorization_pipeline.pkl` contains a **TF-IDF Vectorizer**
- Converts text to 2000-dimensional numerical vector
- Each dimension represents a word's importance

**TF-IDF Process:**
1. **Term Frequency (TF)**: How often word appears in document
2. **Inverse Document Frequency (IDF)**: How rare word is across all documents
3. **TF-IDF Score**: TF × IDF

**Example Vector (simplified):**
```
Word:        "product"  "amaze"  "love"  ... (2000 words)
TF-IDF:      0.5        0.3      0.2    ... (2000 values)
```

**Output Shape:**
```python
vectorized_text.shape  # (1, 2000)
# 1 row (one review), 2000 columns (features)
```

---

## 🔗 Step 7: Combine All Features (Line 449)

### Stack All Features Together

```python
final_features = np.hstack((
    scaled_features,                    # 4 features (scaled numerical)
    np.array([[overall, sentiment_label]]),  # 2 features (rating, sentiment)
    vectorized_text                     # 2000 features (TF-IDF)
))
```

**Feature Combination:**
```
[scaled_features] + [overall, sentiment_label] + [vectorized_text]
[4 features]      + [2 features]                + [2000 features]
= 2006 total features
```

**Example Array:**
```python
final_features = [
    -0.5,    # scaled reviewLength
    0.0,     # scaled helpfulRatio
    0.2,     # scaled wordCount
    -0.1,    # scaled avgWordLength
    5,       # overall rating
    2,       # sentiment_label (positive)
    0.5,     # TF-IDF feature 0
    0.3,     # TF-IDF feature 1
    0.2,     # TF-IDF feature 2
    ...      # ... 1997 more TF-IDF features
]
# Total: 2006 features
```

### Create Final DataFrame (Lines 452-453)

```python
final_df = pd.DataFrame(
    final_features, 
    columns=[
        'reviewLength', 
        'helpfulRatio', 
        'wordCount', 
        'avgWordLength', 
        'overall', 
        'encoded_sentimentLabel'
    ] + [str(i) for i in range(2000)]  # '0', '1', '2', ... '1999'
)
```

**DataFrame Structure:**
```
   reviewLength  helpfulRatio  wordCount  avgWordLength  overall  encoded_sentimentLabel  0    1    2  ...  1999
0         -0.5           0.0        0.2           -0.1        5                      2  0.5  0.3  0.2  ...  0.0
```

---

## 🤖 Step 8: ML Model Prediction (Line 456)

### Load Pre-trained Model

```python
# Model loaded at startup (line 30)
review_classifier_model = joblib.load('Review_classifier_LG.pkl')
```

### Make Prediction

```python
prediction_raw = review_classifier_model.predict(final_df)[0]
```

**What happens inside the model:**
1. **Logistic Regression** receives 2006 features
2. Applies learned weights and bias
3. Calculates probability: `P(FAKE | features)`
4. If probability > 0.5 → `1` (FAKE)
5. If probability ≤ 0.5 → `0` (GENUINE)

**Example:**
```python
# Model calculates:
probability_fake = 0.15  # 15% chance it's fake
probability_genuine = 0.85  # 85% chance it's genuine

# Since 0.15 < 0.5:
prediction_raw = 0  # GENUINE
```

### Get Probabilities (Optional, Lines 458-466)

```python
if hasattr(review_classifier_model, 'predict_proba'):
    proba = review_classifier_model.predict_proba(final_df)[0].tolist()
    # Returns: [0.85, 0.15]  # [GENUINE_prob, FAKE_prob]
```

---

## 🏷️ Step 9: Map to Label (Lines 468-485)

### Convert Number to Text

```python
if label_is_fake_when_one:
    prediction = "FAKE" if prediction_raw == 1 else "GENUINE"
else:
    prediction = "FAKE" if prediction_raw == 0 else "GENUINE"

# Our example: prediction_raw = 0
# → prediction = "GENUINE"
```

**Label Mapping:**
- `0` → `"GENUINE"`
- `1` → `"FAKE"`

---

## 📤 Step 10: Return Result (Line 487)

### Render Template with Prediction

```python
return render_template('result.html', prediction=prediction)
```

**What happens:**
- Flask renders `templates/result.html`
- Passes `prediction = "GENUINE"` to template
- Template displays result to user

**Template receives:**
```html
<h2>{{ prediction }}</h2>
<!-- Renders as: <h2>GENUINE</h2> -->
```

---

## 📊 Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER SUBMITS FORM                                        │
│    review_text = "This product is amazing! I love it!"      │
│    overall = 5                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. FLASK RECEIVES REQUEST                                    │
│    POST /predict                                             │
│    Extract: review_text, overall                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. FEATURE EXTRACTION                                       │
│    • review_length = 35                                     │
│    • word_count = 6                                         │
│    • avg_word_length = 4.0                                  │
│    • sentiment_score = 0.636 (VADER)                        │
│    • sentiment_label = 2 (positive)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. FEATURE SCALING                                          │
│    StandardScaler transforms:                                │
│    [35, 0.0, 6, 4.0] → [-0.5, 0.0, 0.2, -0.1]               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. TEXT PREPROCESSING                                        │
│    "This product is amazing! I love it!"                    │
│    ↓ lowercase                                               │
│    ↓ remove punctuation                                      │
│    ↓ tokenize                                                │
│    ↓ remove stopwords                                        │
│    ↓ lemmatize                                               │
│    → "product amaze love"                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. TEXT VECTORIZATION                                       │
│    TF-IDF Vectorizer:                                        │
│    "product amaze love" → [2000-dim vector]                 │
│    [0.5, 0.3, 0.2, ..., 0.0]                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. COMBINE ALL FEATURES                                     │
│    [4 scaled] + [2 categorical] + [2000 TF-IDF]             │
│    = 2006 total features                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. ML MODEL PREDICTION                                      │
│    Logistic Regression:                                     │
│    Input: 2006 features                                     │
│    Output: probability = 0.15 (FAKE)                        │
│    prediction_raw = 0 (GENUINE)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. MAP TO LABEL                                             │
│    0 → "GENUINE"                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 10. DISPLAY RESULT                                          │
│     Render result.html with prediction="GENUINE"             │
│     User sees: "GENUINE"                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Points

### **Input Flow:**
1. **Form** → `review_text`, `overall` rating
2. **Backend** → Extracts features, preprocesses text
3. **ML Pipeline** → Scales, vectorizes, combines features
4. **Model** → Makes prediction
5. **Output** → "FAKE" or "GENUINE"

### **Feature Engineering:**
- **4 Numerical Features**: Length, word count, avg word length, helpful ratio
- **2 Categorical Features**: Rating, sentiment label
- **2000 Text Features**: TF-IDF vectorization
- **Total: 2006 features** → ML Model

### **ML Model:**
- **Algorithm**: Logistic Regression
- **Input**: 2006 features
- **Output**: Binary classification (0/1)
- **Mapping**: 0 = GENUINE, 1 = FAKE

---

## 💡 Example Walkthrough

**Input:**
```
Review: "This product is amazing! I love it!"
Rating: 5 stars
```

**Processing:**
```
1. Features: length=35, words=6, avg_len=4.0, sentiment=0.636
2. Scaled: [-0.5, 0.0, 0.2, -0.1]
3. Preprocessed: "product amaze love"
4. Vectorized: [2000-dim array]
5. Combined: [2006 features]
6. Prediction: 0 (GENUINE)
```

**Output:**
```
"GENUINE"
```

---

This is the complete flow from form submission to ML prediction! 🚀

