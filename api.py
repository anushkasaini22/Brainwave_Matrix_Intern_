# If uploaded to Colab directly
df = pd.read_csv('/content/INSTAGRAM_REVIEWS.csv')

df.head()

# Select columns and drop missing
df = df[['review_text','review_rating']].dropna()  # Adjust if column names differ
# Binary sentiment: 1 for positive (>3), 0 for negative (≤3)
df['sentiment'] = df['review_rating'].apply(lambda x: 1 if x > 3 else 0)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = clean_text(text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

df['clean_content'] = df['review_text'].apply(preprocess)
df.head()

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_content']).toarray()
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Model trained successfully")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#cm = confusion_matrix(y_test, y_pred)
#plt.figure(figsize=(6,4))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
##plt.ylabel('Actual')
#plt.xlabel('Predicted')
#plt.title('Confusion Matrix')
#plt.show()

pos_words = ' '.join(df[df['sentiment']==1]['clean_content'])
neg_words = ' '.join(df[df['sentiment']==0]['clean_content'])

def plot_top_words(text, title):
    vec = TfidfVectorizer(stop_words='english', max_features=30)
    features = vec.fit_transform([text])
    words = vec.get_feature_names_out()
    scores = features.toarray()[0]
    sorted_words = pd.DataFrame({'Word':words, 'Score':scores}).sort_values('Score',ascending=False)
    plt.figure(figsize=(10,4))
    sns.barplot(x='Word',y='Score',data=sorted_words)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

plot_top_words(pos_words, 'Top Words in Positive Reviews')
plot_top_words(neg_words, 'Top Words in Negative Reviews')

sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.show()

wordcloud_pos = WordCloud(width=500, height=400, background_color='white').generate(' '.join(df[df['sentiment']==1]['clean_content']))
wordcloud_neg = WordCloud(width=500, height=400, background_color='black').generate(' '.join(df[df['sentiment']==0]['clean_content']))

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews')
plt.subplot(1,2,2)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews')
plt.tight_layout()
plt.show()

ef predict_sentiment(rating=None, comment=None):
    """Predict sentiment from rating or comment"""
    if rating is not None:
        try:
            rating = int(rating)
            return 1 if rating > 3 else 0
        except:
            return " Invalid rating. Please enter a number 1–5."
    elif comment is not None and comment.strip() != "":
        cleaned = preprocess(comment)
        vectorized = tfidf.transform([cleaned]).toarray()
        prediction = clf.predict(vectorized)[0]
        return int(prediction)
    else:
        return "Please provide a valid comment or rating."

def interactive_prediction():
    print("\n===== Sentiment Prediction =====")
    print("Choose option:")
    print("1 - Enter a Comment")
    print("2 - Enter a Rating")
…

choice = input("\nYour choice: ").strip().lower()
        if choice == '1':
            comment = input("Enter your comment: ")
            result = predict_sentiment(comment=comment)
            print(f"Prediction: {result} (1=Positive, 0=Negative)")
        elif choice == '2':
            rating = input("Enter rating (1-5): ")
            result = predict_sentiment(rating=rating)
            print(f"Prediction: {result} (1=Positive, 0=Negative)")
        elif choice == '3':
            # Correct column names based on the DataFrame's structure
            row = df.sample(1).iloc[0]
            print(f"\nRandom Dataset Example:\nRating: {row['review_rating']}\nComment: {row['review_text']}")
            pred = predict_sentiment(comment=row['review_text'])
  …
 interactive_prediction()



.....
