from googleapiclient.discovery import build


def fetch_top_videos(api_key, query, max_results=5):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.search().list(
        part="snippet", q=query, type="video", maxResults=max_results, order="relevance"
    )
    response = request.execute()

    videos = []
    for item in response["items"]:
        videos.append(
            {
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
            }
        )

    return videos
