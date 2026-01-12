# ModelData.py
import pandas as pd

def make_daily_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # aggregate per day (across all users, or filter before calling)
    grouped = df.groupby('only_date').agg(
        total_messages=('messages', 'count'),
        total_words=('msg_len_words', 'sum'),
        avg_msg_length=('msg_len_words', 'mean'),
        total_media=('is_media', 'sum'),
        total_links=('link_count', 'sum'),
        avg_sentiment=('sentiment_score', 'mean'),
        avg_emoji_count=('emoji_count', 'mean'),
        weekday=('day_name', lambda x: x.iloc[0])
    ).reset_index()

    # derive numeric weekday
    weekday_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3,
                   'Friday':4, 'Saturday':5, 'Sunday':6}
    grouped['weekday_num'] = grouped['weekday'].map(weekday_map)

    # define busy threshold: e.g. above median of total_messages
    threshold = grouped['total_messages'].median()
    grouped['is_busy'] = (grouped['total_messages'] > threshold).astype(int)

    return grouped, threshold
