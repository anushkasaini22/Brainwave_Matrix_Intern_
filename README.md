# Brainwave_Matrix_Intern_ 
Instagram App Reviews Sentiment Analysis using Python, NLP &amp; ML. Classifies reviews as Positive (1) or Negative (0) with TF-IDF &amp; Naive Bayes. Features 
data cleaning, visualization &amp; interactive prediction in Colab. Built with NLTK, scikit-learn, Matplotlib. Clone, run notebook &amp; explore sentiment trends!


![Instagram Logo](https://github.com/user-attachments/assets/b83a19d1-c4ef-407a-8fd3-384ea0c9b377)

# Objective
The objective of this project was to analyze user reviews of the Instagram app to determine whether the sentiment behind each review is positive or negative.
The goal is to build a machine learning-based text classification model that can take either a written comment or a numeric rating as input and predict sentiment.

# Dataset
The dataset contains Instagram app reviews, each with a rating (1-5 stars) and a text comment.

Examples of negative reviews: “It glitched out and made it super slow... I hate it”

Examples of positive reviews: “Instagram is amazing! Love the new updates”
Dataset link : (https://www.kaggle.com/datasets/bwandowando/3-million-instagram-google-store-reviews?utm)

# Libraries & Tools
Data Processing & Analysis:

pandas, numpy

Visualization:

matplotlib.pyplot, seaborn for plotting sentiment distribution and top words

wordcloud for visualizing most frequent words

Natural Language Processing (NLP):

nltk (stopwords, lemmatization using WordNetLemmatizer)

re (regular expressions for text cleaning)

Machine Learning:

sklearn (CountVectorizer / TfidfVectorizer, Train-Test Split, Classification Models such as Logistic Regression,  Naive Bayes, etc.)

Environment:

Google Colab for code execution and interactive prediction

# Methodology
Step 1 — Data Cleaning & Preprocessing
Removed punctuation, numbers, and special characters using regex (re.sub()).

Converted text to lowercase for uniformity.

Removed stopwords using NLTK stopwords.

Performed lemmatization using WordNetLemmatizer to reduce words to base form.

Step 2 — Exploratory Data Analysis (EDA)
Plotted sentiment distribution using seaborn.countplot.

Generated a Word Cloud to visualize the most frequent words in reviews.

Created Top Words Bar Charts to identify common terms in positive and negative reviews.

Step 3 — Feature Extraction
Used TF-IDF Vectorization to convert text into numerical vectors for model training.

Step 4 — Model Building
Split dataset into training (80%) and testing (20%) sets.

Trained classification models like:

Logistic Regression

Multinomial Naive Bayes

Selected the best-performing model based on accuracy.

Step 5 — Model Evaluation
Evaluated using:

Accuracy

Precision, Recall, F1-score

Confusion Matrix

# Interactive Prediction Feature
An interactive mode was implemented allowing the user to:

Enter a comment → Get predicted sentiment (Positive/Negative)

Enter a rating → Predict sentiment based on rating scale

View a random dataset review with model prediction

Exit interactive mode

 # Results
Model Accuracy: ~85–90% (depending on model choice, e.g., Logistic Regression performed best)

Positive sentiment example:
image : ![Evaluation Accuracy](https://github.com/user-attachments/assets/8425a53a-9018-4065-9831-c1c738c2e358)


Input: "Instagram App" → Prediction: Positive (1)

Negative sentiment example:

Input: "It glitched out... I hate it" → Prediction: Negative (0)

Sentiment distribution showed more positive reviews compared to negative.
                           
