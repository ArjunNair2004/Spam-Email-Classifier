# Spam-Email-Classifier
Classifies spam emails by taking input from the user using NLP

#Python Code
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset and preprocess
try:
    df = pd.read_csv('email50.csv')  # Adjust the file path if necessary
except FileNotFoundError:
    st.error("Dataset not found! Please make sure 'email50.csv' is available.")

X = df.drop(columns=['spam', 'time'])
y = df['spam']
X = pd.get_dummies(X, columns=['number', 'winner'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train or load Naive Bayes model
model_file = 'naive_bayes_spam_model.pkl'
try:
    model = joblib.load(model_file)
    st.success("Loaded pre-trained model.")
except:
    model = MultinomialNB()
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
    st.success("Trained and saved new model.")

# Model accuracy
accuracy = model.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Streamlit user interface
st.title("Email Spam Classifier")
st.write("Classify whether an email is spam based on its characteristics.")

to_multiple = st.radio("Sent to multiple recipients:", ('yes', 'no'))
from_address = st.radio("From a known person:", ('yes', 'no'))
cc = st.radio("Was there a CC:", ('yes', 'no'))
sent_email = st.radio("Is it a sent email:", ('yes', 'no'))
number = st.radio("Number of recipients:", ('small', 'big', 'none'))
winner = st.radio("Does the email mention winning something:", ('yes', 'no'))

if st.button("Classify Email"):
    # Prepare input
    input_data = {
        'to_multiple': [1 if to_multiple == 'yes' else 0],
        'from': [1 if from_address == 'yes' else 0],
        'cc': [1 if cc == 'yes' else 0],
        'sent_email': [1 if sent_email == 'yes' else 0],
        'number_big': [1 if number == 'big' else 0],
        'number_none': [1 if number == 'none' else 0],
        'winner_yes': [1 if winner == 'yes' else 0],
    }

    input_df = pd.DataFrame(input_data)
    for col in X_train.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X_train.columns]

    # Predict
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.write("This email is SPAM")
    else:
        st.write("This email is NOT SPAM")
        
