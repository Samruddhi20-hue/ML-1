import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("autism_detection_model.pkl")

# Define age group mapping
def get_age_group(age):
    if age <= 12:
        return "child"
    elif age <= 18:
        return "teen"
    elif age <= 30:
        return "young"
    elif age <= 50:
        return "adult"
    else:
        return "senior"

# Function to make predictions
def predict_autism(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, age, gender, ethnicity, jaundice, country, used_app, result, relation):
    try:
        # Convert categorical inputs to numerical
        gender = gender.lower()
        jaundice = 1 if jaundice.lower() == "yes" else 0
        used_app = 1 if used_app.lower() == "yes" else 0

        # Compute total score
        total_score = sum([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10])

        # Derive age group
        age_group = get_age_group(age)

        # Ensure correct input format
        input_data = pd.DataFrame([[
            A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, 
            age, gender, ethnicity, jaundice, country, 
            used_app, result, relation, total_score, age_group
        ]], columns=[
            "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
            "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
            "age", "gender", "ethnicity", "jaundice", "contry_of_res",
            "used_app_before", "result", "relation", "total_score", "age_group"
        ])

        # Predict using model
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  

        # Format result
        if prediction == 1:
            return f"(Confidence: {probability:.2%})"
        else:
            return f"(Confidence: {probability:.2%})"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🧩 Autism Prediction System")

    with gr.Row():
        with gr.Column():
            A1 = gr.Slider(0, 1, step=1, label="Social interaction (e.g., eye contact, facial expressions)")
            A2 = gr.Slider(0, 1, step=1, label="Nonverbal communication (e.g., gestures, body language)")
            A3 = gr.Slider(0, 1, step=1, label="Peer relationships (e.g., friendships, cooperative play)")
            A4 = gr.Slider(0, 1, step=1, label="Shared enjoyment (e.g., showing things to others)")
            A5 = gr.Slider(0, 1, step=1, label="Social/emotional reciprocity (e.g., responding to emotions)")

        with gr.Column():
            A6 = gr.Slider(0, 1, step=1, label="Repetitive and Stereotyped Behaviors")
            A7 = gr.Slider(0, 1, step=1, label="Rigid and Inflexible Behaviors")
            A8 = gr.Slider(0, 1, step=1, label="Restricted or Fixated Interests")
            A9 = gr.Slider(0, 1, step=1, label="Sensory Sensitivities or Atypical Sensory Responses")
            A10 = gr.Slider(0, 1, step=1, label="Unusual Use of Objects or Play Behaviors")

    with gr.Row():
        age = gr.Number(label="Age", minimum=1, maximum=100)
        gender = gr.Radio(["Male", "Female"], label="Gender")
        ethnicity = gr.Textbox(label="Ethnicity (e.g., White, Black, Latino)")
        jaundice = gr.Radio(["Yes", "No"], label="Jaundice at Birth")
        country = gr.Textbox(label="Country of Residence")

    with gr.Row():
        used_app = gr.Radio(["Yes", "No"], label="Used Autism Screening App Before?")
        result = gr.Number(label="Test Result Score", minimum=0)
        relation = gr.Dropdown(["Self", "Parent", "Relative", "Others"], label="Relation with the Patient")

    output = gr.Textbox(label="Prediction Result")

    predict_btn = gr.Button("Predict Autism")
    predict_btn.click(
        predict_autism, 
        inputs=[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, age, gender, ethnicity, jaundice, country, used_app, result, relation], 
        outputs=output
    )

# Launch Gradio Apphih
app.launch()
