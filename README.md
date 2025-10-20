# Plant Species Detection using Deep Learning
## Overview

Plant Species Detection is a deep learning project that identifies plant species from an input image using EfficientNetV2B0, a powerful image classification model.
The model is wrapped inside a Streamlit web application for an interactive user experience and deployed on AWS.

 

## Features

 Upload an image of a plant (leaf/flower) for detection.

 Deep learning model based on EfficientNetV2B0 for accurate classification.

 Pretrained model fine-tuned on a custom plant dataset.

 Displays predicted species name and model confidence.

 Deployed on AWS with Streamlit interface.

## Tech Stack
Component	Technology
Framework	TensorFlow / Keras
Model	EfficientNetV2B0
Frontend	Streamlit
Language	Python
Deployment	AWS EC2
Libraries	NumPy, Pandas, Matplotlib, scikit-learn, PIL

## Installation and Setup
Clone the Repository
```
git clone https://github.com/supreet510/Plant-Species-Detection.git
cd Plant_Species_Detection
```
Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```
Install Dependencies
```
pip install -r requirements.txt
```
Run the Application Locally
```
streamlit run app.py
```

App will start at:

http://localhost:8501/



## Example Output

Uploaded Image:

rose_leaf.jpg


Model Output:

    Predicted Species: Juniperus Chinensis (Plumosa Aurea)
    Confidence: 98.7%

Model Performance
    Metric	Value
    Accuracy	96.4%
    Precision	95.8%
    Recall	96.0%
    F1 Score	95.9%

Author

Supreet510
