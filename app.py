from flask import Flask, request, render_template, jsonify
from youtube_api import fetch_top_videos
from sentimental_analysis import analyze_video_sentiments, load_naive_bayes_model
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Load Naive Bayes model (optional)
load_naive_bayes_model()


@app.route("/")
def home():
    """Render home page."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Handle YouTube video sentiment analysis."""
    try:
        # Get the topic and sentiment method from the form data
        topic = request.form.get("topic")
        use_naive_bayes = (
            request.form.get("use_naive_bayes") == "on"
        )  # Checkbox for Naive Bayes

        if not topic:
            return render_template(
                "index.html", error="Please provide a topic to search for."
            )

        # Fetch top YouTube videos related to the topic
        videos = fetch_top_videos(API_KEY, topic)

        if not videos:
            return render_template(
                "index.html", error="No videos found for this topic."
            )

        # Analyze sentiments for the fetched videos
        analysis_results = [
            analyze_video_sentiments(video, use_naive_bayes) for video in videos
        ]

        return render_template("index.html", results=analysis_results)

    except Exception as e:
        return render_template("index.html", error=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
