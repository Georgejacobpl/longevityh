import gradio as gr
import pandas as pd
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

def analyze_supplement(title):
    if not title:
        return "Please enter a video title"
    
    prediction = model.predict(vectorizer.transform([title]))[0]
    confidence = model.predict_proba(vectorizer.transform([title])).max()
    
    result = f"🎯 Sentiment: {prediction.upper()}\n"
    result += f"📊 Confidence: {(confidence*100):.1f}%"
    
    return result

# Create interface
demo = gr.Interface(
    fn=analyze_supplement,
    inputs=gr.Textbox(lines=2, placeholder="Enter supplement video title..."),
    outputs=gr.Textbox(label="Analysis Result"),
    title="💊 Supplement Video Analyzer",
    description="Analyze the sentiment of supplement review videos",
    examples=[
        ["This pre-workout gives me insane energy!"],
        ["Waste of money, doesn't work at all"],
        ["Pretty good supplement for the price"]
    ]
)

demo.launch()
