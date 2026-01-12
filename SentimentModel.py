import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

def make_user_sentiment_dataset(df: pd.DataFrame, selected_user: str) -> pd.DataFrame:
    """COMPLETELY REWRITTEN - NO GROUPBY APPLY"""
    user_df = df[df['users'] == selected_user].copy()
    if len(user_df) < 15:
        return None
    
    daily_stats = []

    for date in user_df['only_date'].unique():
        day_data = user_df[user_df['only_date'] == date]

        if len(day_data) == 0:
            continue

        total_msgs = len(day_data)
        positive_ratio = (day_data['sentiment_label'] == 'Positive').sum() / total_msgs
        avg_words = day_data['msg_len_words'].mean() if 'msg_len_words' in day_data.columns else 5
        avg_emojis = day_data['emoji_count'].mean() if 'emoji_count' in day_data.columns else 0
        avg_hour = day_data['hour'].mean()

        hours = day_data['hour'].dropna()
        if len(hours) > 0:
            peak_hour = int(hours.mode().iloc[0]) if len(hours.mode()) > 0 else int(hours.median())
        else:
            peak_hour = 12

        daily_stats.append({
            'only_date': date,
            'total_messages': total_msgs,
            'positive_ratio': positive_ratio,
            'avg_words_mean': avg_words,
            'avg_emojis': avg_emojis,
            'avg_hour': avg_hour,
            'peak_hour': peak_hour,
            'day_name': day_data['day_name'].iloc[0] if len(day_data) > 0 else 'Monday',
            'month': day_data['month'].iloc[0] if len(day_data) > 0 else 'January'
        })

    if len(daily_stats) < 10:
        return None

    daily_sent = pd.DataFrame(daily_stats)

    median_ratio = daily_sent['positive_ratio'].median()
    daily_sent['is_positive_day'] = (daily_sent['positive_ratio'] > median_ratio).astype(int)

    weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    month_map = {'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5,
                 'July': 6, 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11}

    daily_sent['weekday_num'] = daily_sent['day_name'].map(weekday_map).fillna(0).astype(int)
    daily_sent['month_num'] = daily_sent['month'].map(month_map).fillna(0).astype(int)

    daily_sent['peak_hour'] = daily_sent['peak_hour'].astype(int)

    return daily_sent


def get_best_worst_times(daily_sent: pd.DataFrame, selected_month_num: int = None) -> tuple:
    """Safe version - no int() conversions in loops"""
    positive_days = daily_sent[daily_sent['is_positive_day'] == 1]
    negative_days = daily_sent[daily_sent['is_positive_day'] == 0]

    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    best_times = []
    worst_times = []

    if selected_month_num is not None:
        positive_days = positive_days[positive_days['month_num'] == selected_month_num]
        negative_days = negative_days[negative_days['month_num'] == selected_month_num]

    if len(positive_days) > 0:
        pos_combos = positive_days.groupby(['weekday_num', 'peak_hour']).size().reset_index(name='count')
        top_best = pos_combos.nlargest(3, 'count')
        for _, row in top_best.iterrows():
            best_times.append({
                'weekday': weekday_names[int(row['weekday_num'])],
                'hour': f"{row['peak_hour']:.0f}:00",
                'score': int(row['count'])
            })

    if len(negative_days) > 0:
        neg_combos = negative_days.groupby(['weekday_num', 'peak_hour']).size().reset_index(name='count')
        top_worst = neg_combos.nlargest(3, 'count')
        for _, row in top_worst.iterrows():
            worst_times.append({
                'weekday': weekday_names[int(row['weekday_num'])],
                'hour': f"{row['peak_hour']:.0f}:00",
                'score': int(row['count'])
            })

    return best_times, worst_times


def train_sentiment_model(df: pd.DataFrame, selected_user: str):
    daily_sent = make_user_sentiment_dataset(df, selected_user)
    if daily_sent is None:
        return None, None, None, None, None, None, None

    feature_cols = ['total_messages', 'avg_words_mean', 'avg_emojis', 'avg_hour',
                    'peak_hour', 'weekday_num', 'month_num']

    X = daily_sent[feature_cols].copy()
    for col in feature_cols:
        X[col] = X[col].fillna({
                                   'total_messages': 1,
                                   'avg_words_mean': 5,
                                   'avg_emojis': 0,
                                   'avg_hour': 12,
                                   'peak_hour': 12,
                                   'weekday_num': 0,
                                   'month_num': 0
                               }[col]).astype(float)

    y = daily_sent['is_positive_day'].astype(int)

    if len(X) < 10:
        return None, None, None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, acc, report, daily_sent, feature_cols, daily_sent['positive_ratio'].median()
