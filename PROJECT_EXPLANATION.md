# 📚 Fake Review Detector - Complete Project Explanation

## 🎯 Project Overview

This is a **Machine Learning-powered web application** that detects fake/authentic reviews using Natural Language Processing (NLP) and various text features. The project uses Flask as the web framework, MongoDB for user management, and pre-trained ML models for classification.

---

## 📁 File Structure & Detailed Explanation

### 🔧 **Core Application Files**

#### 1. **`app.py`** - Main Flask Application (516 lines)
**Purpose:** The heart of the application - handles all backend logic, routes, and ML predictions.

**Key Components:**

**A. Initialization & Setup (Lines 1-50)**
- Imports all necessary libraries (Flask, ML libraries, MongoDB, etc.)
- Loads pre-trained models:
  - `scaling_pipeline.pkl` - StandardScaler for numerical features
  - `vectorization_pipeline.pkl` - TF-IDF vectorizer for text
  - `Review_classifier_LG.pkl` - Logistic Regression classifier
- Initializes Flask app, VADER sentiment analyzer, spaCy NLP model
- Sets up MongoDB connection (localhost:27017)
- Configures Flask-Login for authentication

**B. User Management System (Lines 51-250)**
- `User` class: Flask-Login user model with roles (user/admin)
- `load_user()`: Loads user from MongoDB by ID
- `hash_password()` / `check_password()`: Bcrypt password hashing
- `create_default_admin()`: Creates default admin account on startup
  - Username: `admin`
  - Password: `admin123`

**C. Text Preprocessing Functions (Lines 88-124)**
- `expand_contractions()`: Converts "can't" → "cannot", "won't" → "will not", etc.
- `preprocess_and_lemmatize()`: 
  - Converts to lowercase
  - Expands contractions
  - Removes punctuation/numbers
  - Tokenizes text
  - Removes stopwords
  - Lemmatizes words (using spaCy)

**D. Web Scraping Function (Lines 127-228)**
- `scrape_reviews()`: Scrapes product reviews from e-commerce sites
  - **Amazon**: Extracts ASIN, navigates to reviews page, scrapes review text
  - **Flipkart**: Uses CSS selectors to find review elements
  - **Generic sites**: Fallback scraping for other e-commerce platforms
  - Handles pagination, deduplication, and error handling
  - Returns list of review dictionaries

**E. ML Prediction Functions (Lines 252-317)**
- `batch_predict_reviews()`: Processes multiple reviews at once
  - Extracts features: review length, word count, avg word length, sentiment
  - Scales numerical features using pre-trained scaler
  - Preprocesses and vectorizes text
  - Combines all features
  - Makes predictions using Logistic Regression model
  - Returns detailed results with probabilities

**F. Authentication Routes (Lines 320-409)**
- `/login` (GET/POST): User/admin login with role selection
- `/register` (GET/POST): New user registration with validation
- `/logout`: Logout functionality
- `/admin/dashboard`: Admin-only user management page
- `/admin/delete_user/<user_id>`: Delete user (admin only)

**G. Main Application Routes (Lines 411-512)**
- `/` (GET): Home page
- `/form` (GET): Review submission form
- `/predict` (POST): Single review prediction
  - Takes: review text, overall rating (1-5)
  - Extracts features, preprocesses, predicts
  - Returns FAKE or GENUINE
- `/analyze` (GET/POST): Batch URL analysis
  - Scrapes reviews from product URL
  - Analyzes all reviews
  - Shows summary statistics (fake %, genuine %)

---

#### 2. **`requirements.txt`** - Python Dependencies
**Purpose:** Lists all Python packages needed to run the project.

**Key Dependencies:**
- **Flask 3.1.0**: Web framework
- **Flask-Login 0.6.3**: User session management
- **scikit-learn 1.2.2**: Machine learning library
- **spacy 3.8.2**: NLP library for lemmatization
- **nltk 3.9.1**: Natural language toolkit
- **pymongo 4.6.1**: MongoDB driver
- **bcrypt 4.1.2**: Password hashing
- **vaderSentiment 3.3.2**: Sentiment analysis
- **beautifulsoup4 4.12.3**: Web scraping
- **pandas, numpy**: Data manipulation
- **joblib**: Model serialization

---

#### 3. **`README.md`** - Project Documentation
**Purpose:** User-facing documentation explaining the project.

**Contents:**
- Project introduction
- Features overview
- Installation instructions
- Usage guide
- Model information
- Contributing guidelines
- Contact information

---

#### 4. **`SETUP.md`** - Setup Guide
**Purpose:** Detailed setup instructions for developers.

**Key Sections:**
- Prerequisites (Python 3.7+, MongoDB)
- Installation steps
- MongoDB setup (Windows/macOS/Linux)
- Default admin credentials
- Security notes
- Troubleshooting guide

---

### 🎨 **Frontend Template Files** (in `templates/` folder)

#### 5. **`templates/index.html`** - Home Page (350 lines)
**Purpose:** Landing page with project information and navigation.

**Features:**
- **Navigation Bar**: Links to Home, Submit Review, Analyze URL, Login/Register
- **Hero Section**: Welcome message and "Try Now" button
- **How It Works**: Explains the ML process
- **Input Requirements**: Lists what users need to provide
- **Output Explanation**: Describes FAKE vs GENUINE results
- **Responsive Design**: Mobile-friendly layout
- **Background**: Gradient overlay with background image

**Styling:**
- Google Fonts (Roboto)
- CSS animations (fadeIn)
- Modern gradient backgrounds
- Sticky navigation bar

---

#### 6. **`templates/form.html`** - Review Submission Form (260 lines)
**Purpose:** Form for users to submit a single review for analysis.

**Features:**
- **Form Fields:**
  - Review Text (textarea)
  - Overall Rating (1-5 stars, number input)
- **Form Action**: POSTs to `/predict` route
- **Navigation**: Same navbar as other pages
- **Responsive**: Mobile-optimized layout

**User Flow:**
1. User enters review text and rating
2. Submits form
3. Redirected to `/predict` route
4. Results shown on `result.html`

---

#### 7. **`templates/result.html`** - Prediction Result Page (217 lines)
**Purpose:** Displays the prediction result for a single review.

**Features:**
- **Result Display**: Shows "FAKE" or "GENUINE" prominently
- **Action Buttons**: 
  - "Try Another Review" → `/form`
  - "Go to Home" → `/`
- **Clean Design**: Centered result card with clear typography

**Data Shown:**
- Prediction label (FAKE/GENUINE)
- Links to try again or go home

---

#### 8. **`templates/analyze.html`** - URL Analysis Form (76 lines)
**Purpose:** Form to input product URL for batch review analysis.

**Features:**
- **Input Fields:**
  - Product URL (Amazon, Flipkart, etc.)
  - Max reviews to fetch (default: 30, max: 200)
- **Form Action**: POSTs to `/analyze` route
- **Minimal Design**: Simple, focused interface

**User Flow:**
1. User enters product URL
2. Sets max reviews to analyze
3. Submits form
4. Redirected to `analysis.html` with results

---

#### 9. **`templates/analysis.html`** - Batch Analysis Results (108 lines)
**Purpose:** Displays comprehensive results from URL scraping and analysis.

**Features:**
- **Summary Cards:**
  - Total reviews analyzed
  - Number of FAKE reviews (with percentage)
  - Number of GENUINE reviews (with percentage)
- **Detailed Table:**
  - Prediction (FAKE/GENUINE badge)
  - Sentiment score and label
  - Original review text
  - Cleaned/preprocessed text
  - Review metrics (length, word count, avg word length)
- **Color Coding**: 
  - Red badge for FAKE
  - Green badge for GENUINE

**Data Structure:**
Each review shows:
- Original text
- Preprocessed text
- Sentiment analysis
- Feature values
- Prediction result

---

#### 10. **`templates/login.html`** - Login Page (242 lines)
**Purpose:** User and admin authentication page.

**Features:**
- **User Type Selection**: Radio buttons for User/Admin
- **Form Fields:**
  - Username
  - Password
- **Flash Messages**: Shows success/error messages
- **Links**: 
  - Register new account
  - Back to home
- **Modern Design**: Glassmorphism effect with backdrop blur

**Authentication Flow:**
- Checks user type (user/admin)
- Queries appropriate MongoDB collection
- Validates password with bcrypt
- Creates Flask-Login session
- Redirects based on role

---

#### 11. **`templates/register.html`** - Registration Page (211 lines)
**Purpose:** New user account creation.

**Features:**
- **Form Fields:**
  - Username
  - Email
  - Password
  - Confirm Password
- **Validation**: 
  - Password match check
  - Duplicate username/email check
- **Flash Messages**: Success/error notifications
- **Links**: Login page, back to home

**Registration Flow:**
1. User fills form
2. Backend validates inputs
3. Checks for existing users
4. Hashes password with bcrypt
5. Stores in MongoDB `users` collection
6. Redirects to login

---

#### 12. **`templates/admin_dashboard.html`** - Admin Dashboard (352 lines)
**Purpose:** Admin-only page for user management.

**Features:**
- **Statistics Cards:**
  - Total users count
  - Active users count
- **User Management Table:**
  - Username
  - Email
  - Created date
  - Delete action button
- **Delete Functionality**: Confirmation dialog before deletion
- **Flash Messages**: Success/error notifications
- **Access Control**: Only accessible to admin role

**Admin Capabilities:**
- View all registered users
- Delete users
- See user statistics

---

### 🤖 **Machine Learning Model Files**

#### 13. **`Review_classifier_LG.pkl`** - Logistic Regression Model
**Purpose:** Pre-trained classifier that predicts FAKE vs GENUINE.

**Model Details:**
- **Algorithm**: Logistic Regression
- **Input Features**: 
  - Scaled numerical features (review length, word count, etc.)
  - Sentiment features
  - TF-IDF vectorized text (2000 dimensions)
- **Output**: Binary classification (0 = GENUINE, 1 = FAKE)
- **Training**: Trained on Amazon product reviews dataset

**Usage:**
- Loaded at application startup
- Used in `prediction_function()` and `batch_predict_reviews()`
- Makes predictions on preprocessed review data

---

#### 14. **`scaling_pipeline.pkl`** - Feature Scaling Pipeline
**Purpose:** StandardScaler for normalizing numerical features.

**Features Scaled:**
- `reviewLength`: Character count
- `helpfulRatio`: Helpful votes ratio
- `wordCount`: Number of words
- `avgWordLength`: Average word length

**Why Scaling?**
- Ensures all features are on similar scales
- Improves model performance
- Required for Logistic Regression

---

#### 15. **`vectorization_pipeline.pkl`** - TF-IDF Vectorizer
**Purpose:** Converts preprocessed text into numerical vectors.

**Details:**
- **Method**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Output Size**: 2000-dimensional vectors
- **Process**: 
  - Fits on training data vocabulary
  - Transforms new text to same vector space

**Why TF-IDF?**
- Captures word importance
- Reduces impact of common words
- Creates fixed-size feature vectors from variable-length text

---

### 📓 **Data Science Files**

#### 16. **`fake-review-detector.ipynb`** - Jupyter Notebook
**Purpose:** Data science notebook for model development and EDA.

**Contents (from what we can see):**
- Dataset creation from JSON files
- Exploratory Data Analysis (EDA)
- Target variable distribution analysis
- Review length analysis
- Visualization of data patterns

**Key Insights:**
- Genuine:Fake ratio ≈ 1:5
- Shorter reviews more likely genuine
- Longer reviews more likely fake
- Both classes have long-tail distributions

**Note:** This notebook was used during development to:
- Understand the data
- Engineer features
- Train and evaluate models
- Create visualizations

---

### 🗂️ **Data Files**

#### 17. **`nltk_data/`** - NLTK Resources Folder
**Purpose:** Local storage of NLTK data to avoid downloads.

**Contents:**
- **Stopwords**: Word lists for multiple languages (English, Spanish, etc.)
- **Tokenizers**: Punkt tokenizer models for sentence splitting
- **Language Support**: Multiple languages for internationalization

**Why Local?**
- Faster startup (no internet download needed)
- Offline functionality
- Consistent environment

---

### 🌐 **Other Files**

#### 18. **`index.html`** (Root) - Redirect Page (94 lines)
**Purpose:** Landing/redirect page (possibly for deployment).

**Features:**
- Animated countdown timer (70 seconds)
- Redirects to hosted application
- Explains free hosting limitations
- Gradient background animation

**Note:** This appears to be a deployment-specific file, not used in local development.

---

## 🔄 **Application Workflow**

### **Single Review Analysis Flow:**
1. User visits `/form`
2. Enters review text and rating
3. Submits → POST to `/predict`
4. Backend:
   - Preprocesses text
   - Extracts features
   - Scales numerical features
   - Vectorizes text
   - Makes prediction
5. Result displayed on `/result` page

### **Batch URL Analysis Flow:**
1. User visits `/analyze`
2. Enters product URL
3. Submits → POST to `/analyze`
4. Backend:
   - Scrapes reviews from URL
   - Processes each review through ML pipeline
   - Calculates statistics
5. Results displayed on `/analysis` page with table

### **User Authentication Flow:**
1. User registers → stored in MongoDB
2. User logs in → Flask-Login creates session
3. Admin users can access `/admin/dashboard`
4. Regular users can use prediction features

---

## 🧠 **Machine Learning Pipeline**

### **Feature Engineering:**
1. **Text Features:**
   - Review length (characters)
   - Word count
   - Average word length
   - Sentiment score (VADER)
   - Sentiment label (positive/neutral/negative)

2. **Text Preprocessing:**
   - Lowercase conversion
   - Contraction expansion
   - Punctuation removal
   - Stopword removal
   - Lemmatization

3. **Vectorization:**
   - TF-IDF transformation
   - 2000-dimensional feature vectors

4. **Final Feature Set:**
   - 4 scaled numerical features
   - 2 categorical features (overall rating, sentiment label)
   - 2000 TF-IDF features
   - **Total: 2006 features**

### **Model Prediction:**
- Logistic Regression classifier
- Output: Probability scores for FAKE/GENUINE
- Threshold: 1 = FAKE, 0 = GENUINE

---

## 🗄️ **Database Structure (MongoDB)**

### **Collections:**

1. **`users` Collection:**
   ```json
   {
     "_id": ObjectId,
     "username": "string",
     "email": "string",
     "password": "hashed_string",
     "created_at": DateTime
   }
   ```

2. **`admins` Collection:**
   ```json
   {
     "_id": ObjectId,
     "username": "admin",
     "email": "admin@example.com",
     "password": "hashed_string",
     "created_at": DateTime
   }
   ```

---

## 🔐 **Security Features**

1. **Password Hashing**: Bcrypt with salt
2. **Session Management**: Flask-Login secure sessions
3. **Role-Based Access**: Admin routes protected
4. **Input Validation**: Form validation on registration
5. **SQL Injection Protection**: MongoDB parameterized queries

---

## 🚀 **Deployment Considerations**

- **Local Development**: Runs on `localhost:5000`
- **Production**: Requires:
  - MongoDB server
  - Python 3.7+
  - All dependencies installed
  - Model files present
  - NLTK data available

---

## 📊 **Key Technologies Used**

- **Backend**: Flask, Python
- **Frontend**: HTML, CSS, Jinja2 templates
- **Database**: MongoDB
- **ML/NLP**: scikit-learn, spaCy, NLTK, VADER
- **Web Scraping**: BeautifulSoup, requests
- **Authentication**: Flask-Login, bcrypt

---

## 🎯 **Project Strengths**

1. **Complete ML Pipeline**: End-to-end from text to prediction
2. **User Management**: Full authentication system
3. **Batch Processing**: Can analyze multiple reviews
4. **Web Scraping**: Automatic review extraction
5. **Modern UI**: Responsive, attractive design
6. **Admin Features**: User management dashboard

---

## 🔧 **Potential Improvements**

1. **Model Performance**: Could try other algorithms (Random Forest, XGBoost)
2. **Feature Engineering**: Add more features (readability scores, etc.)
3. **Caching**: Cache predictions for same reviews
4. **API**: RESTful API for external integrations
5. **Real-time Updates**: WebSocket for live scraping progress
6. **Export**: Download results as CSV/PDF

---

This project demonstrates a complete **full-stack ML application** with authentication, web scraping, and real-time predictions! 🎉

