import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Set up the page
st.set_page_config(page_title="SMS Scam Classifier", layout="wide")
st.title("📩 SMS Scam vs Trust Classifier")


# Load data
@st.cache_data
def load_data():
    return pd.read_csv("bongo_scam.csv")


df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Overview", "🤖 Train Model", "🧪 Predict"])

# Page 1: Data Overview
if page == "📊 Data Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Category Distribution")
    st.bar_chart(df['Category'].value_counts())

    st.subheader("Word Clouds")
    for label in df['Category'].unique():
        text = " ".join(df[df['Category'] == label]['Sms'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.markdown(f"#### {label.capitalize()} Messages")
        st.image(wordcloud.to_array())

# Page 2: Train Model
elif page == "🤖 Train Model":
    st.subheader("Train and Evaluate Logistic Regression Model")

    X = df['Sms']
    y = df['Category']
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    st.pyplot(fig)

    # Save model
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    st.success("✅ Model and vectorizer saved successfully!")

# Page 3: Predictions
elif page == "🧪 Predict":
    st.subheader("Make a Prediction")

    try:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    except:
        st.error("❌ Model not trained yet. Please go to 'Train Model' first.")
        st.stop()

    msg = st.text_area("Type your message below:")
    if st.button("Classify Message"):
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]
        st.success(f"🧠 Prediction: **{pred.upper()}**")

    st.markdown("---")
    st.subheader("Batch Prediction via File Upload")
    uploaded_file = st.file_uploader("Upload CSV or Excel file (must have 'Sms' column)", type=['csv', 'xlsx'])

    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        if 'Sms' not in data.columns:
            st.error("❌ File must contain a 'Sms' column.")
        else:
            vectors = vectorizer.transform(data['Sms'])
            data['Prediction'] = model.predict(vectors)
            st.success("✅ Predictions completed!")
            st.dataframe(data)
            st.download_button("Download Predictions", data.to_csv(index=False), "predictions.csv")
