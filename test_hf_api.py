# Test script to verify Hugging Face API integration
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Get token from env
hf_token = os.getenv("HUGGINGFACE_API_KEY")

# Initialize Hugging Face client
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-alpha",  # free, open-source chat model
    token=hf_token
)

# Test sentiment analysis
def test_sentiment(text):
    prompt = f"Classify the sentiment of this customer review as exactly one word - either 'Positive', 'Negative', or 'Neutral':\n\nReview: {text}\n\nSentiment:"
    
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=10,
            temperature=0.1,
            return_full_text=False
        )
        print(f"Review: {text}")
        print(f"Response: {response}")
        print(f"Sentiment: {response.strip().split()[0]}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error: {e}")

# Test with sample reviews
if __name__ == "__main__":
    print("Testing Hugging Face API for sentiment analysis...")
    print("=" * 50)
    
    test_reviews = [
        "This product is amazing! I love it.",
        "Terrible quality, waste of money.",
        "It's okay, nothing special."
    ]
    
    for review in test_reviews:
        test_sentiment(review)
