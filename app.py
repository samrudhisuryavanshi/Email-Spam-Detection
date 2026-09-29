import streamlit as st
import joblib
import re

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

# ==========================================
# LOAD MODEL AND TF-IDF VECTORIZER
# ==========================================

try:
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# PAGE TITLE
# ==========================================

st.title("📧 Email Spam Detection")

st.write(
    "Enter an email message below and the machine learning model "
    "will predict whether it is Spam or Not Spam."
)

st.divider()


# ==========================================
# EMAIL INPUT
# ==========================================

message = st.text_area(
    "Enter your email:",
    height=180,
    placeholder="Example: Congratulations! You have won a free prize!"
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("🔍 Check Email", use_container_width=True):

    if not message.strip():
        st.warning("Please enter an email first.")

    else:
        # Clean the email
        cleaned_message = clean_text(message)

        # Convert email to TF-IDF
        message_tfidf = vectorizer.transform([cleaned_message])

        # Make prediction
        prediction = model.predict(message_tfidf)[0]

        # Get probabilities
        probabilities = model.predict_proba(message_tfidf)[0]

        not_spam_probability = probabilities[0]
        spam_probability = probabilities[1]

        st.divider()

        # ==========================================
        # RESULT
        # ==========================================

        if prediction == 1:
            st.error("🚨 SPAM EMAIL")

            st.metric(
                "Spam Probability",
                f"{spam_probability * 100:.2f}%"
            )

        else:
            st.success("✅ NOT SPAM")

            st.metric(
                "Not Spam Probability",
                f"{not_spam_probability * 100:.2f}%"
            )

        # ==========================================
        # PROBABILITY DETAILS
        # ==========================================

        st.subheader("📊 Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Not Spam",
                f"{not_spam_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Spam",
                f"{spam_probability * 100:.2f}%"
            )

        # ==========================================
        # EMAIL SUMMARY
        # ==========================================

        st.subheader("📝 Email Summary")

        word_count = len(message.split())
        character_count = len(message)

        # Count URLs
        links = re.findall(
            r"http\S+|www\S+|https\S+",
            message
        )

        # Count special characters
        special_characters = re.findall(
            r"[^a-zA-Z0-9\s]",
            message
        )

        st.write(f"**Characters:** {character_count}")
        st.write(f"**Words:** {word_count}")
        st.write(f"**Links:** {len(links)}")
        st.write(f"**Special characters:** {len(special_characters)}")

        st.write("**Entered email:**")
        st.info(message)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("ℹ️ About the Project")

st.sidebar.write(
    """
    This project uses Machine Learning to classify
    email messages as Spam or Not Spam.

    **Model:** Logistic Regression

    **Text Features:** TF-IDF

    **Dataset:** SMS Spam Collection

    **Target:**
    - Spam
    - Not Spam
    """
)

st.sidebar.divider()

st.sidebar.write(
    "Developed as a Machine Learning project."
)