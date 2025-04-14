from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load environment variables from .env file
load_dotenv()

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize Naive Bayes components
vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()
naive_bayes_model = None


def fetch_comments(api_key, video_id):
    """Fetch comments for a YouTube video."""
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=100
    )
    response = request.execute()

    comments = [
        item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
        for item in response.get("items", [])
    ]
    return comments


def train_naive_bayes_model(comments, sentiments):
    """Train a Naive Bayes model using comment text and sentiment labels."""
    # Convert sentiments to binary labels: positive = 1, negative = 0
    labels = [1 if sentiment == "positive" else 0 for sentiment in sentiments]

    # Convert comments to feature vectors using TF-IDF
    X = vectorizer.fit_transform(comments)
    X = tfidf_transformer.fit_transform(X)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X, labels)

    return model


def analyze_video_sentiments(video, use_naive_bayes=False):
    """Analyze the sentiment of video comments using VADER or Naive Bayes."""
    comments = fetch_comments(os.getenv("API_KEY"), video["video_id"])
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}

    if use_naive_bayes and naive_bayes_model:
        # Apply Naive Bayes for sentiment classification
        X = vectorizer.transform(comments)
        X = tfidf_transformer.transform(X)
        predictions = naive_bayes_model.predict(X)

        for prediction in predictions:
            if prediction == 1:
                sentiments["positive"] += 1
            else:
                sentiments["negative"] += 1
    else:
        # Use VADER Sentiment Analyzer as default
        for comment in comments:
            sentiment_score = analyzer.polarity_scores(comment)
            if sentiment_score["compound"] >= 0.05:
                sentiments["positive"] += 1
            elif sentiment_score["compound"] <= -0.05:
                sentiments["negative"] += 1
            else:
                sentiments["neutral"] += 1

    total_comments = len(comments) or 1  # Avoid division by zero
    for key in sentiments:
        sentiments[key] = round((sentiments[key] / total_comments) * 100, 2)

    overall = max(sentiments, key=sentiments.get)
    messages = {
        "positive": "Best video! Go ahead 😊",
        "neutral": "It's okay. Might be worth a look 😐",
        "negative": "You are wasting your time 😞",
    }

    return {
        "video": video,
        "sentiments": sentiments,
        "overall": overall,
        "message": messages[overall],
    }


# Optional: Load a pre-trained Naive Bayes model if available
def load_naive_bayes_model():
    """Load a pre-trained Naive Bayes model if available."""
    global naive_bayes_model
    if os.path.exists("naive_bayes_model.pkl"):
        naive_bayes_model = joblib.load("naive_bayes_model.pkl")
