import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load news data from CSV
csv_file_path = 'CNN_Articels_clean.csv'

@st.cache_data(persist=True, show_spinner=False)
def load_news_data():
    try:
        news = pd.read_csv(csv_file_path)
        news['Description'] = news['Description'].fillna('')
        news['Headline'] = news['Headline'].fillna('')
        news['Url'] = news['Url'].fillna('')  # Ensure Url is filled
        return news
    except FileNotFoundError:
        st.error("The CNN_Articels_clean.csv file was not found.")
        st.stop()

news = load_news_data()

# Enhanced preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords and lemmatize words
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join words back into a single string
    preprocessed_text = ' '.join(words)
    return preprocessed_text

# Initialize session state variables
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'user_logged_in' not in st.session_state:
    st.session_state['user_logged_in'] = False
if 'read_history' not in st.session_state:
    st.session_state['read_history'] = []
if 'read_state' not in st.session_state:
    st.session_state['read_state'] = {}
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 0
if 'article_count' not in st.session_state:
    st.session_state['article_count'] = 15
if 'recommendation_page' not in st.session_state:
    st.session_state['recommendation_page'] = 0
if 'num_recommendations' not in st.session_state:
    st.session_state['num_recommendations'] = 10  # Total number of recommendations
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = []
if 'recommendation_evaluation' not in st.session_state:
    st.session_state['recommendation_evaluation'] = {}

# User Authentication Page
def show_login_page():
    st.title("Welcome to NewsTrendz")
    st.subheader("Please sign in to continue")
    username_input = st.text_input("Username", "")
    if st.button("Login"):
        if username_input:
            st.session_state['user_logged_in'] = True
            st.session_state['username'] = username_input
            st.success(f"Welcome, {username_input}!")
        else:
            st.warning("Please enter a username.")

# Display Read History
def show_read_history():
    st.subheader("Your Read History")
    if st.session_state['read_history']:
        for article in st.session_state['read_history']:
            st.markdown(f"<div style='background-color: #d9d9d9; color: black; padding: 10px;margin:2px ;border-radius: 5px;'>- {article}</div>", unsafe_allow_html=True)
    else:
        st.write("No read history available.")

# NDCG and MAP calculations
def ndcg_at_n(recommended, actual, n):
    dcg = sum([1/np.log2(i+2) if rec in actual else 0 for i, rec in enumerate(recommended[:n])])
    idcg = sum([1/np.log2(i+2) for i in range(min(n, len(actual)))] if actual else 0)
    return dcg / idcg if idcg > 0 else 0

def map_at_n(recommended, actual, n):
    ap_sum = 0
    hit_count = 0
    for i, rec in enumerate(recommended[:n]):
        if rec in actual:
            hit_count += 1
            ap_sum += hit_count / (i + 1)
    return ap_sum / min(len(actual), n) if actual else 0

# Evaluate Recommendations
def evaluate_recommendations(recommendations, n=5):
    true_labels = [1 if item in st.session_state['read_history'] else 0 for item in recommendations]
    predicted_labels = [1] * len(recommendations)

    precision = precision_score(true_labels, predicted_labels, zero_division=1)
    recall = recall_score(true_labels, predicted_labels, zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, zero_division=1)

    actual = set(st.session_state['read_history'])
    ndcg_score = ndcg_at_n(recommendations, actual, n)
    map_score = map_at_n(recommendations, actual, n)

    st.session_state['recommendation_evaluation'] = {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'ndcg@{}'.format(n): ndcg_score,
        'map@{}'.format(n): map_score,
    }

# Get hybrid recommendations based on user read history
def get_hybrid_recommendations(user_history, num_recommendations=5):
    news['Processed Text'] = (news['Headline'] + ' ' + news['Description']).apply(preprocess_text)

    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(news['Processed Text'])

    recommended_articles = set()
    indices = pd.Series(news.index, index=news['Headline']).drop_duplicates()

    # Get categories of user-read articles
    user_categories = news[news['Headline'].isin(user_history)]['Category'].unique()

    for title in user_history:
        if title in indices.index:
            idx = indices[title]
            sim_scores = cosine_similarity(tfidf_matrix[idx:idx + 1], tfidf_matrix).flatten()
            top_indices = sim_scores.argsort()[-num_recommendations - 1:-1][::-1]

            # Filter recommendations by category
            for i in top_indices:
                if news['Category'].iloc[i] in user_categories:
                    recommended_articles.add(news['Headline'].iloc[i])

    return list(recommended_articles)[:num_recommendations]

# Main application logic
n = 20

def show_main_interface():
    st.sidebar.header("User Menu")
    st.sidebar.markdown(f"*Logged in as:* {st.session_state['username']}")

    if st.sidebar.button("View Read History"):
        show_read_history()

    if st.sidebar.button("Logout"):
        st.session_state['user_logged_in'] = False
        st.session_state['username'] = ''
        st.session_state['read_history'] = []
        st.success("Logged out successfully.")

    st.title(f"Welcome back, {st.session_state['username']}!")
    st.subheader("Browse the latest news articles")

    search_query = st.text_input("Search for articles", value="", key='search_input', placeholder="Search...")

    # Filter articles based on search query and category
    filtered_articles = news[news['Headline'].str.contains(search_query, case=False, na=False)]

    total_articles = len(filtered_articles)
    total_pages = (total_articles // st.session_state['article_count']) + (1 if total_articles % st.session_state['article_count'] > 0 else 0)

    if total_pages > 0:
        current_page_articles = filtered_articles.iloc[
            st.session_state['current_page'] * st.session_state['article_count'] :
            (st.session_state['current_page'] + 1) * st.session_state['article_count']
        ]

        st.markdown("### Articles Found")
        for index, article in current_page_articles.iterrows():
            color = "#d9d9d9" if article['Headline'] in st.session_state['read_state'] and st.session_state['read_state'][article['Headline']] else "#f0f0f0"
            link_color = "red" if article['Headline'] in st.session_state['read_state'] and st.session_state['read_state'][article['Headline']] else "blue"
            
            st.markdown(f"""
                <div style='background-color: {color}; color: black; padding: 15px; margin-bottom: 10px; border-radius: 5px;'>
                    <strong>{article['Headline']}</strong><br>
                    Published on: {article['Date published']}<br>
                    Category: {article['Category']}<br>
                    {article['Description']}<br>
                    <a href='{article['Url']}' target='_blank' style='color: {link_color};'>Read more</a><br>
                </div>
            """, unsafe_allow_html=True)

            if st.button(f"Mark as Read", key=f"mark_read_{article['Index']}_{index}"):
                st.session_state['read_history'].append(article['Headline'])
                st.session_state['read_state'][article['Headline']] = True

        # Pagination
        if st.session_state['current_page'] > 0:
            if st.button("Previous Page"):
                st.session_state['current_page'] -= 1

        if st.session_state['current_page'] < total_pages - 1:
            if st.button("Next Page"):
                st.session_state['current_page'] += 1

    else:
        st.write("No articles found.")

    # Inside the show_main_interface function

# After the article listing and pagination section
if st.button("Generate Recommendations"):
    user_history = st.session_state['read_history']
    if user_history:  # Only generate recommendations if there are articles in history
        recommendations = get_hybrid_recommendations(user_history, num_recommendations=5)
        st.session_state['recommendations'] = recommendations
        evaluate_recommendations(recommendations)
    else:
        st.warning("Please read some articles to generate recommendations.")

# Display Recommendations
    if st.session_state['recommendations']:
        st.subheader("Recommended Articles")
        for article in st.session_state['recommendations']:
            recommended_article = news[news['Headline'] == article]
            if not recommended_article.empty:
                article_info = recommended_article.iloc[0]
                st.markdown(f"""
                    <div style='background-color: #d9d9d9; color: black; padding: 15px; margin-bottom: 10px; border-radius: 5px; display: flex; justify-content: space-between; align-items: center;'>
                        <strong>{article_info['Headline']}</strong><br>
                            <a href='{article_info['Url']}' target='_blank' style='color: blue;'>Read more</a>
                     </div>
                """, unsafe_allow_html=True)

    else:
        st.write("No recommendations available.")


    # Display evaluations
    if st.session_state['recommendation_evaluation']:
        st.subheader("Recommendation Evaluation")
        for metric, score in st.session_state['recommendation_evaluation'].items():
            st.write(f"{metric}: {score:.4f}")

# Run the app
if st.session_state['user_logged_in']:
    show_main_interface()
else:
    show_login_page()
