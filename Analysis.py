from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

def fetch_stats(selected_user,df):
    url_extractor = URLExtract()
    if selected_user == "Overall":
        total_messages = df['users'].shape[0]
        total_words = 0
        total_links = 0
        for m in df['messages'].tolist():
            links = url_extractor.find_urls(m)
            total_links += len(links)
            m = m.split()
            total_words += len(m)
        total_media = df[df['messages'] == '<Media omitted>'].shape[0]
        return total_messages, total_words,total_media, total_links
    else:
        user_data = df[df['users'] == selected_user]
        total_messages = user_data.shape[0]
        total_words = 0
        total_links = 0
        for m in df[df['users'] == selected_user]['messages'].tolist():
            links = url_extractor.find_urls(m)
            total_links += len(links)
            m = m.split()
            total_words += len(m)
        total_media = user_data[user_data['messages'] == '<Media omitted>'].shape[0]
        return total_messages, total_words,total_media, total_links


def most_busy_users(df):
    user_contribution = df['users'].value_counts().head()
    new_df = round(df['users'].value_counts()/df.shape[0] * 100)
    new_df = new_df.reset_index().rename(columns = {"users" : "Users","count":"Percentage"})

    return user_contribution, new_df

def remove_stop_words(selected_user,df):
    stopwords_file = open("stopwords.txt", 'r')
    stopwords = stopwords_file.read()

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    temp = df[df['users'] != "Group Notification"]
    temp = temp[temp['messages'] != "<Media omitted>"]

    words = []
    for message in temp['messages']:
        for word in message.lower().split():
            if len(word) > 2 and word not in stopwords:
                words.append(word)

    return words

def create_wordcloud(selected_user,df):
    filtered_words = remove_stop_words(selected_user, df)
    filtered_words = " ". join(filtered_words)

    wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color = "white")
    df_wc = wc.generate(filtered_words)

    return df_wc

def most_common_words(selected_user,df):
    filtered_words = remove_stop_words(selected_user,df)

    common_words_df = pd.DataFrame(Counter(filtered_words).most_common(20))
    return common_words_df

def emoji_analysis(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    emojis = []
    for message in df['messages']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    monthly_timeline_df = df.groupby(['year', 'month_num', 'month']).count()['messages'].reset_index()
    time_period = []

    for i in range(monthly_timeline_df.shape[0]):
        time_period.append(f"{monthly_timeline_df['month'][i]} - {monthly_timeline_df['year'][i]}")

    monthly_timeline_df['time_period'] = time_period
    return monthly_timeline_df

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    daily_timeline_df = df.groupby(['only_date']).count()['messages'].reset_index()

    return daily_timeline_df

def weekly_activity(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    return df['day_name'].value_counts()

def monthly_activity(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns= 'period', values= 'messages', aggfunc='count').fillna(0)
    return user_heatmap