import gradio as gr
import joblib
import numpy as np

model = joblib.load("student_dropout_model.pkl")

def predict_dropout(
        age,
        gender,
        year_of_study,
        attendence,
        study_hours,
        previous_gpa,
        backlogs,
        financial_stress,
        stress_level,
        burnout_level
):
    gender = 1 if gender == "male" else 0
    burnout_mapping ={
        "Low" : 0, 
        "Medium" :1,
        "High": 2   }
    
    burnout= burnout_mapping[burnout_level]
    input_data = np.array([[
        age,
        gender,
        year_of_study,
        attendence,
        study_hours,
        previous_gpa,
        backlogs,
        financial_stress,
        stress_level,
        burnout
    ]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        return "Student will drop out"
    else:
        return "Student will not drop out"
    
student_dropout_app = gr.Interface(
    fn = predict_dropout,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(["Male","Female"],label = "Gender"),
        gr.Dropdown([1,2,3,4], label="Year of Study"),
        gr.Slider(0,100,label="Attendence percentage"),
        gr.Number(label="study hours per day"),
        gr.Number(label="previous gpa"),
        gr.Number(label="backlogs"),
        gr.Slider(1,10, step = 1,label="financial stress score"),
        gr.Slider(1,10,step = 1, label =" stress level"),
        gr.Dropdown(["Low","Medium","High"],label = " burnout level"),
    ],
    outputs = gr.Textbox(label = "prediction"),
    title =" Student Dropout Risk Prediction",
    description= "predict whether a student has High or Low dropout risk"
)
student_dropout_app.launch()