import re
import pandas as pd
import emoji
from urlextract import URLExtract

def preprocessing(data):
    pattern = r"\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s?[a-zA-Z]{2}"

    messages = re.split(pattern, data)[1:]
    messages = [m.replace("-", "").strip() for m in messages]

    date = re.findall(pattern, data)
    date = [d.replace(",", "").replace("\u202f", " ") for d in date]

    dataset = pd.DataFrame({
        'userMessage': messages,
        'date': date
    })

    dataset['date'] = pd.to_datetime(date)

    users = []
    msg = []

    for m in dataset['userMessage'].tolist():
        parts = m.split(':', 1)

        if len(parts) == 2:
            users.append(parts[0].strip())

            # Get the message and strip it
            content = parts[1].strip()

            # If content is empty string, label it; otherwise keep it
            msg.append(content if content else f"{parts[0].strip()} Initiated a Call")

        else:
            users.append("Group Notification")
            msg.append(m)

    dataset['users'] = users
    dataset['messages'] = msg
    # number of characters
    dataset['msg_len_chars'] = dataset['messages'].apply(len)
    # number of words
    dataset['msg_len_words'] = dataset['messages'].apply(lambda x: len(x.split()))

    def count_emojis(text):
        return sum(1 for c in text if c in emoji.EMOJI_DATA)

    dataset['emoji_count'] = dataset['messages'].apply(count_emojis)

    extractor = URLExtract()

    dataset['is_media'] = dataset['messages'].apply(lambda x: x == '<Media omitted>')
    dataset['link_count'] = dataset['messages'].apply(lambda x: len(extractor.find_urls(x)))
    dataset['has_link'] = dataset['link_count'] > 0

    dataset.drop(columns='userMessage', inplace=True)

    only_date = dataset['date'].dt.date
    year = dataset['date'].dt.year
    month = dataset['date'].dt.month_name()
    month_num = dataset['date'].dt.month
    day_name = dataset['date'].dt.day_name()
    day = dataset['date'].dt.day
    hour = dataset['date'].dt.hour
    minute = dataset['date'].dt.minute

    dataset['only_date'] = only_date
    dataset['year'] = year
    dataset['month_num'] = month_num
    dataset['month'] = month
    dataset['day_name'] = day_name
    dataset['day'] = day
    dataset['hour'] = hour
    dataset['minute'] = minute

    period = []
    for hour in dataset[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    dataset['period'] = period

    # Add weekday_num for Engagement Predictor
    dataset['weekday_num'] = dataset['day_name'].map({
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }).fillna(0).astype(int)

    # Add month_num for month filtering
    dataset['month_num'] = dataset['month'].map({
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }).fillna(1).astype(int)

    return dataset