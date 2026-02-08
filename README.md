#  Plant Species Detection using Deep Learning

Plant Species Detection is a deep learning project that identifies plant species from an uploaded image. The project uses a fine-tuned **EfficientNetV2B0** model and provides an interactive web interface built with **Streamlit**. The application can be deployed locally and on cloud platforms such as **AWS EC2**.

---
## Live Link
```
http://15.134.212.209:8501/
```
---
---
## Dataset Link
```
https://tinyurl.com/48kpzk2k
```
---

##  Features

-  Upload an image of a plant leaf or plant
-  Deep Learning model using EfficientNetV2B0
-  Displays predicted plant species with confidence score
-  Interactive web interface using Streamlit
-  Deployable on AWS EC2 or locally
-  Fast and easy to use

---

##  Tech Stack

- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow / Keras  
- **Model Architecture:** EfficientNetV2B0  
- **Frontend:** Streamlit  
- **Libraries:** NumPy, Matplotlib
- **Deployment:** AWS EC2  

---

##  Project Structure

Plant-Species-Detection/
     app.py # Streamlit application

     pipeline.py # Prediction pipeline

     plant_species_model.keras # Trained deep learning model

     class_names.txt # Class labels

     plant_project_final.ipynb # Training & experimentation notebook

     requirements.txt # Project dependencies

     README.md # Project documentation


---

##  How It Works

1. User uploads an image through the Streamlit web interface.
2. The image is preprocessed (resized).
3. The trained EfficientNetV2B0 model predicts the plant species.
4. The predicted class name and confidence score are displayed to the user.

---

##  Installation & Setup

###  Clone the repository

```bash
git clone https://github.com/supreet510/Plant-Species-Detection.git
cd Plant-Species-Detection
 Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
 Install dependencies
pip install -r requirements.txt
 Run the Streamlit app
streamlit run app.py
Open your browser and go to:

http://localhost:8501
 Example Output
Input: Plant Image
Output:

Predicted Species: Juniperus Chinensis (Plumosa Aurea)
Confidence: 98.7%
```
## Model Details
```
Architecture: EfficientNetV2B0

Transfer learning used for faster training and better accuracy

Trained on a custom plant species dataset

Saved as .keras model file
```

## Deployment (AWS EC2)

```
Launch an EC2 instance

Install Python, pip, and required libraries

Clone the repository

Open port 8501 in the security group

Run the Streamlit app using:

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Access the app using:

http://15.134.212.209:8501/
```
 
## Author
Supreet Kaur
