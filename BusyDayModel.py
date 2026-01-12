import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from ModelData import make_daily_dataset

def train_busy_day_model(df: pd.DataFrame):
    daily_df, threshold = make_daily_dataset(df)

    feature_cols = [
        'total_words',
        'avg_msg_length',
        'total_media',
        'total_links',
        'avg_sentiment',
        'avg_emoji_count',
        'weekday_num'
    ]
    X = daily_df[feature_cols]
    y = daily_df['is_busy']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)

    return model, acc, report, threshold, daily_df, feature_cols


def get_busy_best_worst(daily_df):
    """Month-aware best/worst days"""
    best_days = []
    worst_days = []
    
    monthly_stats = daily_df.groupby(['month_num', 'weekday_num']).agg({
        'total_messages': ['mean', 'count']
    }).reset_index()
    
    monthly_stats.columns = ['month_num', 'weekday_num', 'avg_msgs', 'day_count']
    monthly_stats['msg_bonus'] = monthly_stats['avg_msgs'] - daily_df['total_messages'].median()
    
    top_busy = monthly_stats.nlargest(4, 'avg_msgs')
    weekday_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    
    for _, row in top_busy.iterrows():
        best_days.append({
            'weekday': weekday_names[int(row['weekday_num'])],
            'msg_bonus': f"+{row['msg_bonus']:.0f}",
            'avg_msgs': row['avg_msgs']
        })
    
    bottom_quiet = monthly_stats.nsmallest(4, 'avg_msgs')
    for _, row in bottom_quiet.iterrows():
        worst_days.append({
            'weekday': weekday_names[int(row['weekday_num'])],
            'msg_count': f"{row['avg_msgs']:.0f} avg",
            'day_count': row['day_count']
        })
    
    return best_days, worst_days
