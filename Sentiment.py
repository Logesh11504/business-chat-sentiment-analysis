from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

_analyzer = SentimentIntensityAnalyzer()

def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # works in-place but also returns df for convenience
    scores = df['messages'].apply(lambda x: _analyzer.polarity_scores(str(x))['compound'])
    df['sentiment_score'] = scores

    def label(score):
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_label'] = df['sentiment_score'].apply(label)
    return df
