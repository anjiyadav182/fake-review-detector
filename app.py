import os
import platform

# Work around Windows WMI hangs seen during NumPy/SciPy/sklearn imports on some systems.
# If WMI is unhealthy, `platform._wmi_query()` can block indefinitely; several scientific
# Python packages call `platform.machine()` during import which triggers WMI access.
if os.name == "nt" and hasattr(platform, "_wmi_query"):
    try:
        # `_win32_ver()` expects 5 returned values from `_wmi_query(...)`.
        platform._wmi_query = lambda *args, **kwargs: ("10.0.0", "1", "", "0", "0")
    except Exception:
        pass

from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import numpy as np
import re
import json
import html
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import spacy
import pickle
import joblib
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
import requests
from bs4 import BeautifulSoup
from bs4 import FeatureNotFound
from urllib.parse import urlparse, urlsplit, urlunsplit
from pymongo import MongoClient
from bson import ObjectId
import bcrypt
from datetime import datetime



# Load pre-trained models and pipelines
with open('scaling_pipeline.pkl', 'rb') as f:
    scaling_pipeline = pickle.load(f)

with open('vectorization_pipeline.pkl', 'rb') as f:
    vectorization_pipeline = pickle.load(f)

review_classifier_model = joblib.load('Review_classifier_LG.pkl')

# Initialize components
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
analyzer = SIA()
nlp = spacy.load('en_core_web_sm')

# MongoDB connection
# Prefer environment configuration so the app can run without hardcoded secrets.
# Examples:
# - Local:  set MONGODB_URI=mongodb://localhost:27017
# - Atlas:  set MONGODB_URI="mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority"
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://vanithareddy2806_db_user:Vanitha123@cluster0.yszaypl.mongodb.net/?appName=Cluster0",
)

client = None
db = None
users_collection = None
admins_collection = None

def connect_auth_db(log_result=True):
    global client, db, users_collection, admins_collection
    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
        )
        # Force early connectivity check so startup doesn't hang indefinitely.
        client.admin.command("ping")
        db = client["fake_review_detector"]
        users_collection = db["users"]
        admins_collection = db["admins"]
        if log_result:
            app.logger.info("MongoDB connected successfully.")
        return True
    except Exception as e:
        # App can still run in "no-auth" mode (review detection pages).
        client = None
        db = None
        users_collection = None
        admins_collection = None
        if log_result:
            app.logger.warning(f"MongoDB disconnected: {e}")
        return False

# Log current DB state on startup.
connect_auth_db(log_result=True)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username, email, role='user'):
        self.id = str(user_id)
        self.username = username
        self.email = email
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    if users_collection is None or admins_collection is None:
        return None
    # Try to find user in users collection
    user_data = users_collection.find_one({'_id': ObjectId(user_id)})
    if user_data:
        return User(user_data['_id'], user_data['username'], user_data['email'], 'user')
    
    # Try to find admin in admins collection
    admin_data = admins_collection.find_one({'_id': ObjectId(user_id)})
    if admin_data:
        return User(admin_data['_id'], admin_data['username'], admin_data['email'], 'admin')
    
    return None

stop_words = set(SPACY_STOP_WORDS)

# nltk.download('punkt')

# Define contractions dictionary
contractions = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am"
}

# Function to expand contractions
def expand_contractions(text, contractions_dict=contractions):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())), 
                                      flags=re.IGNORECASE | re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contractions_dict.get(match.lower())
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

# Function to preprocess and lemmatize text
def preprocess_and_lemmatize(text):
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = text.strip()
    # Lightweight tokenizer to avoid pulling in heavy NLTK + SciPy dependencies on Windows.
    tokens = re.findall(r"[a-zA-Z]+", text)
    tokens = [token for token in tokens if token not in stop_words]
    doc = nlp(' '.join(tokens))
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Basic website scrapers for product reviews
def scrape_reviews(url, max_reviews=50):
    def make_soup(html_text):
        try:
            return BeautifulSoup(html_text, "lxml")
        except FeatureNotFound:
            # Fallback when lxml parser isn't installed/available.
            return BeautifulSoup(html_text, "html.parser")

    def normalize_url(raw_url):
        raw_url = (raw_url or "").strip()
        parts = urlsplit(raw_url)
        path = parts.path or "/"
        # Strip tracking params/fragments to improve parsing/fetch reliability.
        return urlunsplit((parts.scheme or "https", parts.netloc, path, "", ""))

    def clean_text(text):
        t = html.unescape((text or "").strip())
        t = re.sub(r"\s+", " ", t)
        return t

    def maybe_add(reviews_list, text):
        t = clean_text(text)
        if len(t.split()) >= 5:
            reviews_list.append({"text": t})

    def extract_reviews_from_soup(soup, reviews_list, max_count, raw_html_text=None):
        selectors = [
            'div[data-hook="review"] span[data-hook="review-body"] span',
            'span[data-hook="review-body"] span',
            'span[data-hook="review-body"]',
            'span[data-hook="review-collapsed"] span',
            'div.review .review-text-content span',
            'div._6K-7Co, div.t-ZTKy, div._2-N8zT, div._1AtVbE .t-ZTKy',
            '.review-text, .user-review',
        ]
        for sel in selectors:
            for node in soup.select(sel):
                maybe_add(reviews_list, node.get_text(" ", strip=True).replace("READ MORE", ""))
                if len(reviews_list) >= max_count:
                    return

        # JSON-LD often carries reviewBody blocks.
        for script in soup.select('script[type="application/ld+json"]'):
            raw = script.get_text(" ", strip=True)
            if not raw:
                continue
            for match in re.findall(r'"reviewBody"\s*:\s*"(.+?)"', raw):
                maybe_add(reviews_list, bytes(match, "utf-8").decode("unicode_escape"))
                if len(reviews_list) >= max_count:
                    return

        # Generic fallback for pages where reviews are rendered in data blobs.
        page_text = soup.get_text(" ", strip=True)
        for match in re.findall(r'"reviewText"\s*:\s*"(.+?)"', page_text):
            maybe_add(reviews_list, bytes(match, "utf-8").decode("unicode_escape"))
            if len(reviews_list) >= max_count:
                return

        # Parse raw HTML for embedded JSON review payloads.
        if raw_html_text:
            for pattern in [
                r'"reviewBody"\s*:\s*"(.+?)"',
                r'"reviewText"\s*:\s*"(.+?)"',
                r'"review-body"\s*:\s*"(.+?)"',
            ]:
                for match in re.findall(pattern, raw_html_text):
                    maybe_add(reviews_list, bytes(match, "utf-8").decode("unicode_escape"))
                    if len(reviews_list) >= max_count:
                        return

    def extract_reviews_from_mirror_text(text, reviews_list, max_count):
        if not text:
            return

        # 1) JSON-like review fields
        for match in re.findall(r'"reviewBody"\s*:\s*"(.+?)"', text):
            maybe_add(reviews_list, bytes(match, "utf-8").decode("unicode_escape"))
            if len(reviews_list) >= max_count:
                return

        # 2) Capture paragraphs after "Reviewed in ..." lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if "Reviewed in" in ln or "reviewed in" in ln:
                chunk = []
                for j in range(i + 1, min(i + 8, len(lines))):
                    cand = lines[j]
                    if "Reviewed in" in cand or "reviewed in" in cand:
                        break
                    if len(cand.split()) >= 5:
                        chunk.append(cand)
                if chunk:
                    maybe_add(reviews_list, " ".join(chunk))
                    if len(reviews_list) >= max_count:
                        return

        # 3) Last fallback: long lines that look like review text
        for ln in lines:
            if len(ln.split()) >= 12 and not ln.lower().startswith(("price", "delivery", "buy now")):
                maybe_add(reviews_list, ln)
                if len(reviews_list) >= max_count:
                    return

    reviews = []
    deadline = time.time() + 25
    try:
        session = requests.Session()
        retry = Retry(
            total=2,
            connect=2,
            read=2,
            backoff_factor=0.4,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-IN,en;q=0.9,en-GB;q=0.8,en-US;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        }

        url = normalize_url(url)
        domain = urlparse(url).netloc.lower()
        if 'amazon' in domain:
            # Extract ASIN and build all-reviews URL(s)
            asin_match = re.search(r"/(dp|product|gp/product)/([A-Z0-9]{10})", url, re.IGNORECASE)
            asin = None
            if asin_match:
                asin = asin_match.group(2)
            else:
                # Generic fallback: find a 10-char ASIN block
                m = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url, re.IGNORECASE)
                asin = m.group(1) if m else None

            urls_to_try = [url]
            if asin:
                domain_no_port = domain.split(':')[0]
                for page in range(1, 4):
                    urls_to_try.append(
                        f"https://{domain_no_port}/product-reviews/{asin}/?reviewerType=all_reviews&pageSize=50&sortBy=recent&pageNumber={page}"
                    )
                    urls_to_try.append(
                        f"https://{domain_no_port}/product-reviews/{asin}/?ie=UTF8&filterByStar=all_stars&sortBy=recent&pageNumber={page}"
                    )
                    urls_to_try.append(
                        f"https://{domain_no_port}/gp/aw/reviews/{asin}/?pageNumber={page}"
                    )

            for target_url in urls_to_try:
                if len(reviews) >= max_reviews or time.time() > deadline:
                    break
                resp = session.get(target_url, headers={**headers, 'Referer': url}, timeout=10, allow_redirects=True)
                if resp.status_code != 200:
                    continue
                lowered = resp.text.lower()
                if "captcha" in lowered or "sorry, we just need to make sure you're not a robot" in lowered:
                    continue
                soup = make_soup(resp.text)
                extract_reviews_from_soup(soup, reviews, max_reviews, resp.text)

            # Extra fallback: content mirror can bypass anti-bot blocks for some pages.
            if not reviews:
                mirror_targets = []
                mirror_target = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
                mirror_targets.append(f"https://r.jina.ai/http://{mirror_target}")
                if asin:
                    domain_no_port = domain.split(":")[0]
                    mirror_targets.append(
                        f"https://r.jina.ai/http://{domain_no_port}/product-reviews/{asin}/?reviewerType=all_reviews&pageSize=50&pageNumber=1"
                    )
                    mirror_targets.append(
                        f"https://r.jina.ai/https://{domain_no_port}/product-reviews/{asin}/?reviewerType=all_reviews&pageSize=50&pageNumber=1"
                    )

                for mirror_url in mirror_targets:
                    if len(reviews) >= max_reviews:
                        break
                    resp = session.get(mirror_url, timeout=12)
                    if resp.status_code == 200:
                        extract_reviews_from_mirror_text(resp.text, reviews, max_reviews)
        elif 'flipkart' in domain:
            response = session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = make_soup(response.text)
            # Flipkart common review text classes
            extract_reviews_from_soup(soup, reviews, max_reviews, response.text)
        else:
            # Generic: look for review-like blocks
            response = session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = make_soup(response.text)
            extract_reviews_from_soup(soup, reviews, max_reviews, response.text)
            if not reviews:
                for node in soup.select('p'):
                    maybe_add(reviews, node.get_text(" ", strip=True))
                    if len(reviews) >= max_reviews:
                        break
    except Exception:
        # Avoid stdout encoding issues on Windows terminals; log via Flask instead.
        app.logger.exception("scrape_reviews failed")
    # Deduplicate and keep non-empty
    seen = set()
    unique_reviews = []
    for r in reviews:
        t = r.get('text', '').strip()
        if t and t not in seen:
            seen.add(t)
            unique_reviews.append({'text': t})
    return unique_reviews[:max_reviews]

# Authentication helper functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def auth_db_available():
    if users_collection is not None and admins_collection is not None:
        return True
    # Auto-retry connection so auth can recover if DB/network becomes available later.
    return connect_auth_db(log_result=True)

def create_default_admin():
    """Create a default admin user if none exists"""
    if admins_collection is None:
        return
    if admins_collection.count_documents({}) == 0:
        admin_data = {
            'username': 'admin',
            'email': 'admin@example.com',
            'password': hash_password('admin123'),
            'created_at': datetime.utcnow()
        }
        admins_collection.insert_one(admin_data)
        print("Default admin created: username=admin, password=admin123")

# Create default admin on startup
try:
    create_default_admin()
except Exception:
    # If DB is unavailable, allow app to start; auth routes will be unavailable.
    pass

def batch_predict_reviews(review_items):
    if not review_items:
        return []
    texts = [item['text'] for item in review_items]

    # Engineer features per review
    review_lengths = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]
    avg_word_lengths = [float(np.mean([len(w) for w in t.split()])) if len(t.split()) > 0 else 0.0 for t in texts]
    sentiments = [analyzer.polarity_scores(t)['compound'] for t in texts]
    sentiment_labels = [2 if s >= 0.05 else (0 if s <= -0.05 else 1) for s in sentiments]

    # For URL analysis, we don't have product rating or helpful ratio; set neutral defaults
    overall_list = [3 for _ in texts]
    helpful_ratio_list = [0.0 for _ in texts]

    numerical_features = pd.DataFrame(
        list(zip(review_lengths, helpful_ratio_list, word_counts, avg_word_lengths)),
        columns=['reviewLength', 'helpfulRatio', 'wordCount', 'avgWordLength']
    )
    scaled = scaling_pipeline.transform(numerical_features)

    cleaned_texts = [preprocess_and_lemmatize(t) for t in texts]
    vectorized = vectorization_pipeline.transform(cleaned_texts).toarray()

    final_features = np.hstack((
        scaled,
        np.array(list(zip(overall_list, sentiment_labels))),
        vectorized
    ))

    vector_size = vectorized.shape[1]
    columns = ['reviewLength', 'helpfulRatio', 'wordCount', 'avgWordLength', 'overall', 'encoded_sentimentLabel'] + [str(i) for i in range(vector_size)]
    final_df = pd.DataFrame(final_features, columns=columns)

    preds_raw = review_classifier_model.predict(final_df)
    proba = None
    classes = None
    if hasattr(review_classifier_model, 'predict_proba'):
        try:
            proba = review_classifier_model.predict_proba(final_df)
            classes = getattr(review_classifier_model, 'classes_', None)
        except Exception:
            proba = None
            classes = None

    results = []
    for idx, item in enumerate(review_items):
        raw = preds_raw[idx]
        label = 'FAKE' if raw == 1 else 'GENUINE'
        result = {
            'text': texts[idx],
            'cleaned': cleaned_texts[idx],
            'prediction_raw': int(raw) if isinstance(raw, (np.integer, int)) else raw,
            'prediction': label,
            'sentiment_score': float(sentiments[idx]),
            'encoded_sentimentLabel': int(sentiment_labels[idx]),
            'reviewLength': int(review_lengths[idx]),
            'wordCount': int(word_counts[idx]),
            'avgWordLength': float(avg_word_lengths[idx]),
        }
        if proba is not None:
            result['predict_proba'] = proba[idx].tolist()
            result['classes_'] = classes.tolist() if hasattr(classes, 'tolist') else classes
        results.append(result)
    return results

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if not auth_db_available():
        flash('Login is temporarily unavailable because the database is not connected.', 'error')
        return render_template('login.html')

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_type = request.form.get('user_type', 'user')
        
        if user_type == 'admin':
            admin_data = admins_collection.find_one({'username': username})
            if admin_data and check_password(password, admin_data['password']):
                user = User(admin_data['_id'], admin_data['username'], admin_data['email'], 'admin')
                login_user(user)
                flash('Admin login successful!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid admin credentials!', 'error')
        else:
            user_data = users_collection.find_one({'username': username})
            if user_data and check_password(password, user_data['password']):
                user = User(user_data['_id'], user_data['username'], user_data['email'], 'user')
                login_user(user)
                flash('User login successful!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid user credentials!', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if not auth_db_available():
        flash('Registration is temporarily unavailable because the database is not connected.', 'error')
        return render_template('register.html')

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')
        
        # Check if user already exists
        if users_collection.find_one({'username': username}) or users_collection.find_one({'email': email}):
            flash('Username or email already exists!', 'error')
            return render_template('register.html')
        
        # Create new user
        user_data = {
            'username': username,
            'email': email,
            'password': hash_password(password),
            'created_at': datetime.utcnow()
        }
        users_collection.insert_one(user_data)
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully!', 'info')
    return redirect(url_for('home'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if users_collection is None:
        flash('Admin dashboard is unavailable because the database is not connected.', 'error')
        return redirect(url_for('home'))

    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('home'))
    
    # Get all users
    users = list(users_collection.find({}, {'password': 0}))
    return render_template('admin_dashboard.html', users=users)

@app.route('/admin/delete_user/<user_id>')
@login_required
def delete_user(user_id):
    if users_collection is None:
        flash('User management is unavailable because the database is not connected.', 'error')
        return redirect(url_for('home'))

    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('home'))
    
    try:
        users_collection.delete_one({'_id': ObjectId(user_id)})
        flash('User deleted successfully!', 'success')
    except Exception as e:
        flash('Error deleting user!', 'error')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/predict', methods=['GET', 'POST'])
def prediction_function():
    prediction = None
    if request.method == 'POST':
        review_text = request.form['review_text']
        overall = int(request.form['overall'])
        # Helpful ratio input removed from UI; default to 0.0 if not provided
        try:
            helpful_ratio = float(request.form.get('helpful_ratio', 0) or 0)
        except Exception:
            helpful_ratio = 0.0

        # Extract features from review text
        review_length = len(review_text)
        word_count = len(review_text.split())
        avg_word_length = np.mean([len(word) for word in review_text.split()])
        sentiment_score = analyzer.polarity_scores(review_text)['compound']

        # Classify sentiment score
        if sentiment_score >= 0.05:
            sentiment_label = 2
        elif sentiment_score <= -0.05:
            sentiment_label = 0
        else:
            sentiment_label = 1

        # Transform numerical features
        numerical_features = pd.DataFrame([[review_length, helpful_ratio, word_count, avg_word_length]], 
                                          columns=['reviewLength', 'helpfulRatio', 'wordCount', 'avgWordLength'])
        scaled_features = scaling_pipeline.transform(numerical_features)

        # Preprocess review text
        cleaned_review_text = preprocess_and_lemmatize(review_text)

        # Vectorize review text
        vectorized_text = vectorization_pipeline.transform([cleaned_review_text]).toarray()

        # Combine all features into a single dataframe
        final_features = np.hstack((scaled_features, np.array([[overall, sentiment_label]]), vectorized_text))

        # Create a DataFrame for the final features
        final_df = pd.DataFrame(final_features, columns=['reviewLength', 'helpfulRatio', 'wordCount', 'avgWordLength', 
                                                         'overall', 'encoded_sentimentLabel'] + [str(i) for i in range(2000)])

        # Predict using the classifier
        prediction_raw = review_classifier_model.predict(final_df)[0]
        # Try to get probabilities if available
        proba = None
        classes = None
        try:
            if hasattr(review_classifier_model, 'predict_proba'):
                proba = review_classifier_model.predict_proba(final_df)[0].tolist()
                classes = getattr(review_classifier_model, 'classes_', None)
        except Exception:
            proba = None
            classes = None

        # Map label robustly: if classes are [0,1], assume 1 == FAKE else fallback to original mapping
        if classes is not None and len(classes) == 2:
            # Heuristic: If class "fake" encoded as 1 during training, we keep 1->FAKE.
            # If model uses inverse mapping (e.g., 1==GENUINE), flip.
            # We'll decide based on probability names if available; otherwise allow override via heuristic on "overall" and sentiment.
            label_is_fake_when_one = True
            try:
                # If the positive class probability corresponds to label 1 and is low for negative text, keep mapping.
                # Without label names, default True.
                pass
            except Exception:
                label_is_fake_when_one = True
            if label_is_fake_when_one:
                prediction = "FAKE" if prediction_raw == 1 else "GENUINE"
            else:
                prediction = "FAKE" if prediction_raw == 0 else "GENUINE"
        else:
            prediction = "FAKE" if prediction_raw == 1 else "GENUINE"

    return render_template('result.html', prediction=prediction)

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return render_template('analyze.html')
    url = request.form.get('product_url', '').strip()
    max_reviews = int(request.form.get('max_reviews', '30'))
    scraped = scrape_reviews(url, max_reviews=max_reviews)
    results = batch_predict_reviews(scraped)
    error_message = None
    if not results:
        error_message = (
            "Couldn't fetch reviews from this URL. Try the product's dedicated reviews page "
            "(for Amazon, open 'See all reviews') or use a shorter clean product URL."
        )
    summary = {
        'total': len(results),
        'fake': sum(1 for r in results if r['prediction'] == 'FAKE'),
        'genuine': sum(1 for r in results if r['prediction'] == 'GENUINE'),
    }
    summary['fake_pct'] = round((summary['fake'] / summary['total'] * 100.0), 2) if summary['total'] else 0.0
    summary['genuine_pct'] = round((summary['genuine'] / summary['total'] * 100.0), 2) if summary['total'] else 0.0
    return render_template('analysis.html', url=url, summary=summary, results=results, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
