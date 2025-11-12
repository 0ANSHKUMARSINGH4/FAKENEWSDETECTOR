import feedparser

# Trusted global + Indian news sources
RSS_SOURCES = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "Reuters": "http://feeds.reuters.com/reuters/topNews",
    "CNN": "http://rss.cnn.com/rss/edition.rss",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "NYTimes": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "NDTV": "https://feeds.feedburner.com/ndtvnews-top-stories",
    "The Hindu": "https://www.thehindu.com/feeder/default.rss",
    "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "India Today": "https://www.indiatoday.in/rss/1206578",
}

def fetch_latest_news(limit=10):
    """Fetch latest headlines from multiple sources."""
    headlines = []
    per_feed = max(1, limit // max(1, len(RSS_SOURCES)))
    for source, url in RSS_SOURCES.items():
        try:
            feed = feedparser.parse(url)
            entries = getattr(feed, "entries", [])[:per_feed]
            for e in entries:
                title = getattr(e, "title", None)
                if title:
                    headlines.append((title.strip(), source))
        except Exception:
            continue
    # Deduplicate
    seen = set()
    deduped = []
    for t, s in headlines:
        if t not in seen:
            deduped.append((t, s))
            seen.add(t)
    return deduped[:limit]
