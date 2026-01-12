import streamlit as st
import pandas as pd
from Preprocessor import preprocessing
from Analysis import fetch_stats, most_busy_users, create_wordcloud, most_common_words, emoji_analysis, \
    monthly_timeline, daily_timeline, weekly_activity, monthly_activity, activity_heatmap
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from Sentiment import add_sentiment
from BusyDayModel import train_busy_day_model, get_busy_best_worst
from SentimentModel import train_sentiment_model, get_best_worst_times
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import os

st.set_page_config(
    layout="wide"
)

if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False
if 'last_file_id' not in st.session_state:
    st.session_state.last_file_id = None

st.sidebar.title("Business Chat Sentiment Analyzer")

st.sidebar.markdown("""
    ### 💡 Tip:
    **Upload any exported whatsapp chat.** **Ensure the filename has NO emojis.** *Example: `chat.txt` ✅*
""")

uploaded_file = st.sidebar.file_uploader("Choose a file", type=['txt'])
DEFAULT_FILE_PATH = "WhatsApp Chat with SanDiya_Nandhinee Developments.txt" 

is_filename_safe = True
if uploaded_file:
    try:
        uploaded_file.name.encode('ascii') 
        file_identifier = f"{uploaded_file.size}_{uploaded_file.type}"
    except UnicodeEncodeError:
        st.sidebar.error("❌ ERROR: Filename contains emojis!")
        st.sidebar.info("Please rename the file on your computer and upload again.")
        is_filename_safe = False
        file_identifier = "invalid"
else:
    file_identifier = "default_demo"

if st.session_state.last_file_id != file_identifier:
    st.session_state.last_file_id = file_identifier
    st.session_state.show_analysis = False
    st.cache_data.clear()
    if is_filename_safe:
        st.rerun()

@st.cache_data(show_spinner="Processing chat...")
def get_data(file_source, is_path=False):
    try:
        if is_path:
            with open(file_source, "rb") as f:
                bytes_data = f.read()
        else:
            file_source.seek(0) 
            bytes_data = file_source.read() 
        
        data = bytes_data.decode('utf-8', errors='replace')
        df = preprocessing(data)
        df = add_sentiment(df)
        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

dataset = None

if uploaded_file is not None and is_filename_safe:
    dataset = get_data(uploaded_file)
    st.sidebar.success("✅ Your file loaded successfully!")

elif os.path.exists(DEFAULT_FILE_PATH):
    dataset = get_data(DEFAULT_FILE_PATH, is_path=True)
    
    st.sidebar.info("📂 **Demo Chat Uploaded**")
    st.sidebar.caption("Currently showing data from the default demo file. Upload your own `.txt` file above to analyze your chats.")
    
    if not st.session_state.show_analysis:
        st.session_state.show_analysis = True

if dataset is not None:
    user_list = dataset['users'].unique().tolist()
    if 'Group Notification' in user_list:
        user_list.remove('Group Notification')
    user_list.insert(0, 'Overall')

    selected_user = st.sidebar.selectbox("Show Analysis Wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        st.session_state.show_analysis = True

    if st.session_state.show_analysis:
        st.title(f"Analysis for {selected_user}")
        st.dataframe(dataset)
        
        st.title("Top Statistics")
        col1,col2,col3,col4 = st.columns(4)
        messages, words, media,links = fetch_stats(selected_user,dataset)

        with col1:
            st.header("Total Message")
            st.title(messages)

        with col2:
            st.header("Total Words")
            st.title(words)

        with col3:
            st.header("Total Media")
            st.title(media)

        with col4:
            st.header("Total Links")
            st.title(links)

        st.title("Sentiment Overview")

        if selected_user != 'Overall':
            sent_df = dataset[dataset['users'] == selected_user]
        else:
            sent_df = dataset

        sent_counts = sent_df['sentiment_label'].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Counts")
            st.bar_chart(sent_counts)

        with col2:
            st.subheader("Average Sentiment Over Time")
            sent_time = sent_df.groupby('only_date')['sentiment_score'].mean().reset_index()
            fig, ax = plt.subplots()
            ax.plot(sent_time['only_date'], sent_time['sentiment_score'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig)






        st.title("Engagement Predictor")
        if selected_user != "Overall":
            user_data = dataset[dataset['users'] == selected_user]
            days_available = user_data['only_date'].nunique()
            if days_available >= 10:
                
                WEEKDAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                
                if st.button("🔍 Analyze Patterns"):
                    with st.spinner("Calculating engagement..."):
                        model_df = user_data if selected_user != 'Overall' else dataset
                        
                        model_df = model_df.copy()
                        weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                                    'Friday': 4, 'Saturday': 5, 'Sunday': 6}
                        model_df['weekday_num'] = model_df['day_name'].map(weekday_map).fillna(0).astype(int)
                        model_df['month_num'] = pd.to_datetime(model_df['only_date']).dt.month
                        
                        daily_stats = model_df.groupby(['only_date', 'weekday_num', 'month_num']).agg({
                            'messages': 'count',
                            'msg_len_words': 'mean',
                            'emoji_count': 'mean'
                        }).reset_index()
                        
                        daily_stats['engagement_score'] = (
                            daily_stats['messages'] * 0.5 +
                            daily_stats['msg_len_words'] * 0.3 + 
                            daily_stats['emoji_count'] * 0.2
                        )
                        
                        weekday_overall = daily_stats.groupby('weekday_num')['engagement_score'].mean().reset_index()
                        weekday_overall['day_name'] = [WEEKDAY_NAMES[int(i)] for i in weekday_overall['weekday_num']]
                        
                        st.session_state.engagement_top_days = weekday_overall.nlargest(3, 'engagement_score').to_dict('records')
                        st.session_state.engagement_bottom_days = weekday_overall.nsmallest(3, 'engagement_score').to_dict('records')
                        st.session_state.daily_stats = daily_stats
                        st.session_state.WEEKDAY_NAMES = WEEKDAY_NAMES
                        st.session_state.engagement_analyzed = True
                        st.rerun()
                
                if st.session_state.get('engagement_analyzed', False):
                    month_names = ['Overall', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    selected_view = st.selectbox("View Patterns:", month_names)
                    
                    top_days = st.session_state.engagement_top_days
                    bottom_days = st.session_state.engagement_bottom_days
                    daily_stats = st.session_state.daily_stats
                    weekday_names = st.session_state.get('WEEKDAY_NAMES', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                    
                    if selected_view != 'Overall':
                        month_num = month_names.index(selected_view)
                        month_data = daily_stats[daily_stats['month_num'] == month_num]
                        
                        if len(month_data) > 0:
                            month_weekday = month_data.groupby('weekday_num')['engagement_score'].mean().reset_index()
                            month_weekday['day_name'] = [weekday_names[int(i)] for i in month_weekday['weekday_num']]
                            
                            top_days = month_weekday.nlargest(3, 'engagement_score').to_dict('records')
                            bottom_days = month_weekday.nsmallest(3, 'engagement_score').to_dict('records')
                        else:
                            top_days = []
                            bottom_days = []
                            st.info(f"📭 **{selected_view}:** No data")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### 🏆 **BEST 3 DAYS**")
                        if top_days:
                            for i, day_data in enumerate(top_days, 1):
                                score = day_data['engagement_score']
                                st.markdown(f"**{i}.** {day_data['day_name']} **{score:.1f}** ⭐⭐⭐")
                        else:
                            st.info("No data")
                    
                    with col2:
                        st.markdown("### 😴 **WORST 3 DAYS**")
                        if bottom_days:
                            for i, day_data in enumerate(bottom_days, 1):
                                score = day_data['engagement_score']
                                st.markdown(f"**{i}.** {day_data['day_name']} **{score:.1f}** ⭐")
                        else:
                            st.info("No data")
                    
                    with st.form("enhanced_scenario"):
                        st.subheader("🔮 Smart Scenario Tester")
                        s_col1, s_col2 = st.columns(2)
                        with s_col1:
                            test_month_name = st.selectbox("Month:", month_names[1:])
                            test_day_num = st.selectbox("Day:", range(7), 
                                                    format_func=lambda x: weekday_names[x])
                        with s_col2:
                            test_msgs = st.slider("Messages:", 1, 100, 20)
                            test_words = st.slider("Words/Msg:", 1, 30, 8)
                            test_emojis = st.slider("Emojis/Msg:", 0.0, 3.0, 0.5)
                        
                        test_submit = st.form_submit_button("🎯 Predict", use_container_width=True)

                    if test_submit:
                        test_month_num = month_names.index(test_month_name)
                        test_day_name = weekday_names[test_day_num]
                        
                        activity_score = test_msgs * 0.5 + test_words * 0.3 + test_emojis * 0.2
                        
                        month_data = st.session_state.daily_stats[st.session_state.daily_stats['month_num'] == test_month_num]
                        month_modifier = 1.0
                        if len(month_data) > 0:
                            month_day_avg = month_data[month_data['weekday_num'] == test_day_num]['engagement_score'].mean()
                            month_overall_avg = month_data['engagement_score'].mean()
                            if not pd.isna(month_day_avg):
                                month_modifier = 1.3 if month_day_avg > month_overall_avg else 0.8
                        
                        day_match_top = any(day['day_name'] == test_day_name for day in st.session_state.engagement_top_days)
                        day_match_bottom = any(day['day_name'] == test_day_name for day in st.session_state.engagement_bottom_days)
                        day_modifier = 1.2 if day_match_top else (0.85 if day_match_bottom else 1.0)
                        
                        final_score = activity_score * month_modifier * day_modifier
                        
                        col1, col2, col3 = st.columns([2,1,1])
                        with col1:
                            if final_score > 28:
                                st.markdown("### 🚀 **PERFECT TIMING**")
                                st.success(f"**{final_score:.0f}/100**")
                                st.balloons()
                            else:
                                st.markdown("### 📉 **LOW ENGAGEMENT**")
                                st.warning(f"**{final_score:.0f}/100**")
                        
                        with col2:
                            st.metric("Activity", f"{activity_score:.0f}")
                        with col3:
                            st.metric("History", f"{month_modifier*day_modifier:.0%}")
                        
                        st.markdown("### **Score Breakdown:**")
                        st.metric("📨 Activity", f"{activity_score:.0f}", f"{activity_score:.0f}")
                        st.metric("📅 Month History", f"{month_modifier:.0%}", f"{month_modifier:.0%}")
                        st.metric("📊 Day History", f"{day_modifier:.0%}", f"{day_modifier:.0%}")
                        st.metric("🎯 FINAL", f"{final_score:.0f}", "✅" if final_score > 28 else "❌")

                        
            else:
                st.warning(f"Need 10+ Days of Chat History ({days_available} available)")
        
        else:
            st.warning(f"Select a Specific User with 10+ Days of Chat History")




        st.title("Smart Approach Timing Optimizer")

        weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                        'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        month_map = {'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5,
                        'July': 6, 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11}


        @st.cache_data
        def cached_train_sentiment(user_name, user_data):
            return train_sentiment_model(user_data, user_name)

        if selected_user != 'Overall':
            user_data = dataset[dataset['users'] == selected_user]
            days_available = user_data['only_date'].nunique()
            
            if days_available >= 15:
                if st.button("🚀 Train Model"):
                    with st.spinner("Analyzing sentiment patterns..."):
                        result = cached_train_sentiment(selected_user, dataset)
                        if result[0] is not None:
                            st.session_state.s_model = result[0]
                            st.session_state.s_acc = result[1]
                            st.session_state.s_daily = result[3]
                            st.session_state.sent_model_trained = True
                            st.rerun()

                if st.session_state.get('sent_model_trained', False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Accuracy", f"{st.session_state.s_acc:.1%}")
                        st.metric("Days Analyzed", len(st.session_state.s_daily))
                    with col2:
                        st.caption("**What the model learned:**")
                        st.caption("• Message volume (35%)")
                        st.caption("• Message length (25%)")
                        st.caption("• Emojis & timing (40%)")

                    if st.button("📊 Show What Drives Sentiment"):
                        importances = st.session_state.s_model.feature_importances_
                        features = ['Messages', 'Words/Msg', 'Emojis', 'Avg Hour', 'Peak Hour', 'Weekday', 'Month']

                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = plt.cm.viridis(importances)
                        ax.barh(features, importances, color=colors)
                        ax.set_title(f"📈 {selected_user}'s Sentiment Drivers")
                        ax.set_xlabel("Importance")
                        for i, v in enumerate(importances):
                            ax.text(v + 0.01, i, f'{v:.0%}', va='center')
                        st.pyplot(fig)

                    st.markdown("---")
                    selected_month = st.selectbox("Filter by Month:", ['All'] + list(month_map.keys()))
                    month_filter = month_map[selected_month] if selected_month != 'All' else None

                    best_times, worst_times = get_best_worst_times(st.session_state.s_daily, month_filter)

                    row1_col1, row1_col2 = st.columns(2)
                    with row1_col1:
                        st.subheader("🏆 TOP 3 BEST TIMES")
                        if best_times:
                            for i, slot in enumerate(best_times[:3], 1):
                                st.markdown(f"**{i}.** {slot['weekday']} **{slot['hour']}**")
                                st.caption(f"({slot['score']} positive days)")
                        else:
                            st.info("🔍 Need more positive days...")

                    with row1_col2:
                        st.subheader("❌ TOP 3 WORST TIMES")
                        if worst_times:
                            for i, slot in enumerate(worst_times[:3], 1):
                                st.error(f"**{i}.** {slot['weekday']} **{slot['hour']}**")
                                st.caption(f"({slot['score']} negative days)")
                        else:
                            st.info("🔍 Need more negative days...")

                    st.markdown("---")
                    with st.form("approach_engine"):
                        st.subheader("🔮 Scenario Predictor")

                        input_col1, input_col2 = st.columns(2)
                        with input_col1:
                            day = st.selectbox("📅 Day:", list(weekday_map.keys()))
                            month_input = st.selectbox("📆 Month:", list(month_map.keys()))
                            msg_count = st.slider("📨 Messages:", 1, 100, 20)

                        with input_col2:
                            peak_hr = st.slider("⏰ Peak Hour:", 0, 23, 19)
                            avg_words = st.slider("✍️ Words/Message:", 1, 30, 8)
                            avg_emojis = st.slider("😊 Emojis/Message:", 0.0, 5.0, 0.5)

                        if st.form_submit_button("🎯 Predict Best Time?", use_container_width=True):

                            features = [[
                                float(msg_count),  
                                float(avg_words),  
                                float(avg_emojis),  
                                float(peak_hr),  
                                float(peak_hr),  
                                float(weekday_map[day]),  
                                float(month_map[month_input])  
                            ]]

                            model = st.session_state.s_model
                            prediction = model.predict(features)[0]
                            probabilities = model.predict_proba(features)[0]

                            pred_time_str = f"{day[:3]} {peak_hr:02d}:00"
                            is_best_match = any(
                                slot['weekday'][:3] == day[:3] and slot['hour'][:2] == f"{peak_hr:02d}"
                                for slot in best_times
                            )
                            is_worst_match = any(
                                slot['weekday'][:3] == day[:3] and slot['hour'][:2] == f"{peak_hr:02d}"
                                for slot in worst_times
                            )

                            ml_confidence = probabilities[1]
                            historical_boost = 1.15 if is_best_match else (0.85 if is_worst_match else 1.0)
                            final_score = ml_confidence * historical_boost

                            col1, col2, col3 = st.columns([2, 1, 1])

                            with col1:
                                if final_score > 0.6:
                                    st.markdown("### 🎉 **APPROACH NOW!**")
                                    st.success(f"**Confidence: {final_score:.1%}**")
                                    st.balloons()
                                else:
                                    st.markdown("### ⏳ **WAIT**")
                                    st.error(f"**Confidence: {final_score:.1%}**")

                            with col2:
                                st.metric("ML Score", f"{ml_confidence:.0%}")

                            with col3:
                                st.metric("Final", "✅ GO" if final_score > 0.6 else "❌ WAIT")

                            st.markdown("---")
                            st.subheader("**Prediction Logic**")
                            explanation = []

                            if msg_count > 25:
                                explanation.append("✅ **High volume** = engaged mood")
                            if avg_words > 12:
                                explanation.append("✅ **Long messages** = talkative")
                            if avg_emojis > 1.0:
                                explanation.append("✅ **Many emojis** = positive")

                            if is_best_match:
                                explanation.append("🎯 **Perfect historical timing!**")
                            elif is_worst_match:
                                explanation.append("⚠️ Risky timing (overridden by activity)")

                            for point in explanation[:3]:
                                st.success(point)

                            if not explanation:
                                st.info("🧠 Model using learned patterns")

                    with st.expander("ℹ️ How it works"):
                        st.markdown("""
                        **Model learns from your chat patterns:**
                        1. **High message volume** → User is engaged/happy
                        2. **Long messages** → Talkative/positive mood  
                        3. **Emojis** → Emotional/positive
                        4. **Timing patterns** → User's best/worst hours

                        **Even "worst times" can be good** if activity is high!
                        """)
            else:
                st.warning(f"Need 15+ Days of Chat History ({days_available} days available)")

        else:
            st.warning(f"Select a Specific User with 15+ Days of Chat History")




        st.title("Monthly Timeline")
        timeline_df = monthly_timeline(selected_user, dataset)
        fig, ax = plt.subplots()
        ax.plot(timeline_df['time_period'], timeline_df['messages'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title("Daily Timeline")
        timeline_df = daily_timeline(selected_user, dataset)
        fig, ax = plt.subplots()
        ax.plot(timeline_df['only_date'], timeline_df['messages'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.title("Most Busiest Month")
            month_activity_df = monthly_activity(selected_user, dataset)
            fig, ax = plt.subplots()
            ax.barh(month_activity_df.index, month_activity_df.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.title("Most Busiest Day")
            week_activity_df = weekly_activity(selected_user, dataset)
            fig, ax = plt.subplots()
            ax.barh(week_activity_df.index, week_activity_df.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = activity_heatmap(selected_user,dataset)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)


        if selected_user == 'Overall':
            x, new_df = most_busy_users(dataset)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                st.title('Most Busy Users')
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.title("Busy User Stats")
                st.dataframe(new_df)

        st.title("Word Cloud")
        word_cloud = create_wordcloud(selected_user, dataset)
        fig,ax = plt.subplots()
        ax.imshow(word_cloud)
        st.pyplot(fig)

        most_common_df = most_common_words(selected_user, dataset)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)


        st.title("Frequently Used Emojis")
        emoji_df = emoji_analysis(selected_user,dataset).rename(columns={0 :"Emoji", 1: "Count"})
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            fig = px.pie(emoji_df.head(), values=emoji_df['Count'].head(), names=emoji_df['Emoji'].head())
            st.plotly_chart(fig)