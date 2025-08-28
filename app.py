# import packages
import streamlit as st
import pandas as pd
import os
import plotly.express as px
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests
from textblob import TextBlob


# Load environment variables
load_dotenv()

# Get Hugging Face token from environment
hf_token = os.getenv("HUGGINGFACE_API_KEY")

# Choose a sentiment classification model (returns negative / neutral / positive labels)
HF_SENTIMENT_MODEL = os.getenv("HF_SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")

if not hf_token:
    st.warning("Hugging Face API token not found in environment. Set HUGGINGFACE_API_KEY in .env.")

# Client dedicated to sentiment classification
sentiment_client = InferenceClient(
    model=HF_SENTIMENT_MODEL,
    token=hf_token
)


# Helper function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    return csv_path


# Function to get sentiment using Hugging Face API with TextBlob fallback
@st.cache_data
def get_sentiment(text):
    if not text or pd.isna(text):
        return "Neutral"
    
    # Try Hugging Face API first (if token available)
    if hf_token:
        try:
            # Call classification endpoint
            result = sentiment_client.text_classification(text)
            
            # Handle different response formats
            if isinstance(result, list) and result:
                if isinstance(result[0], dict):
                    # Standard format: [{"label": "POSITIVE", "score": 0.99}, ...]
                    best = max(result, key=lambda r: r.get("score", 0))
                    label = best.get("label", "neutral").lower()
                elif isinstance(result[0], list):
                    # Nested format: [[{"label": "POSITIVE", "score": 0.99}]]
                    best = max(result[0], key=lambda r: r.get("score", 0))
                    label = best.get("label", "neutral").lower()
                else:
                    label = "neutral"
                
                # Normalize labels - this model uses star ratings (1-5 stars)
                if "1" in label or "2" in label:
                    return "Negative"
                elif "4" in label or "5" in label:
                    return "Positive"
                elif "3" in label:
                    return "Neutral"
                
                # Fallback to standard sentiment labels
                if any(word in label for word in ["positive", "pos"]):
                    return "Positive"
                elif any(word in label for word in ["negative", "neg"]):
                    return "Negative"
                else:
                    return "Neutral"
                    
        except Exception as e:
            # If any error with HuggingFace, fall through to TextBlob
            pass
    
    # Fallback to TextBlob (offline sentiment analysis)
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Map polarity to sentiment categories
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"
            
    except Exception as e:
        return "Neutral"




st.title("🔍 GenAI Sentiment Analysis Dashboard")
st.write("This is your GenAI-powered data processing app.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Load Dataset"):
        try:
            csv_path = get_dataset_path()
            df = pd.read_csv(csv_path)
            st.session_state["df"] = df.head(10)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🔍 Analyze Sentiment"):
        if "df" in st.session_state:
            try:
                with st.spinner("Analyzing sentiment..."):
                    # Show progress
                    progress_bar = st.progress(0)
                    df_copy = st.session_state["df"].copy()
                    
                    # Process each row with progress
                    for i, row in enumerate(df_copy.iterrows()):
                        idx, data = row
                        sentiment = get_sentiment(data["SUMMARY"])
                        df_copy.at[idx, "Sentiment"] = sentiment
                        progress_bar.progress((i + 1) / len(df_copy))
                    
                    st.session_state["df"] = df_copy
                    progress_bar.empty()
                    st.success("Sentiment analysis completed!")
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.write("Error details:", e)
        else:
            st.warning("Please load the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)

    # Visualization using Plotly if sentiment analysis has been performed
    if "Sentiment" in st.session_state["df"].columns:
        st.subheader(f"📊 Sentiment Breakdown for {product}")
        
        # Create Plotly bar chart for sentiment distribution using filtered data
        sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        # Define custom order and colors
        sentiment_order = ['Negative', 'Neutral', 'Positive']
        sentiment_colors = {'Negative': 'red', 'Neutral': 'lightgray', 'Positive': 'green'}
        
        # Only include sentiment categories that actually exist in the data
        existing_sentiments = sentiment_counts['Sentiment'].unique()
        filtered_order = [s for s in sentiment_order if s in existing_sentiments]
        filtered_colors = {s: sentiment_colors[s] for s in existing_sentiments if s in sentiment_colors}
        
        # Reorder the data according to our custom order (only for existing sentiments)
        sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=filtered_order, ordered=True)
        sentiment_counts = sentiment_counts.sort_values('Sentiment')
        
        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title=f"Distribution of Sentiment Classifications - {product}",
            labels={"Sentiment": "Sentiment Category", "Count": "Number of Reviews"},
            color="Sentiment",
            color_discrete_map=filtered_colors
        )
        fig.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Number of Reviews",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

