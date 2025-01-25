
# Facial Expression Recognition Project  

This repository contains a facial expression recognition application built using deep learning and deployed via **Streamlit**. The project aims to predict facial expressions such as happiness, sadness, anger, and more from uploaded images.  

## Features  

- **Pre-trained Deep Learning Model**: A convolutional neural network built with TensorFlow for accurate emotion detection.  
- **Image Processing**: Automatically converts images to grayscale and resizes them to the required format.  
- **Interactive Web Application**: User-friendly Streamlit app for image upload and prediction.  

## Try the App  

You can try the live version of this app here: [Facial Expression Recognition App](https://facial-expression-recognition-hamdii.streamlit.app/)  

## Installation  

1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/hamdii10/facial-expression-recognition.git  
   cd facial-expression-recognition  
   ```  

2. **Set up a virtual environment** (optional but recommended):  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  # For Linux/macOS  
   venv\Scripts\activate     # For Windows  
   ```  

3. **Install dependencies**:  
   ```bash  
   pip install -r requirements.txt  
   ```  

## Project Structure  

```  
facial-expression-recognition/  
├── models/                                  # Pre-trained model files  
│   └── model.keras                          # TensorFlow model for facial expression recognition  
├── app/                                     # Application scripts  
│   └── app.py                               # Streamlit application script  
├── notebooks/                               # Jupyter notebooks for analysis and training  
│   └── Facial_Expression_Recognition.ipynb  # Notebook for model training and evaluation  
├── README.md                                # Project overview and instructions  
├── LICENSE                                  # License file  
└── requirements.txt                         # Python dependencies for the project  
```  

## How to Run  

1. **Start the Streamlit app**:  
   ```bash  
   streamlit run app/app.py  
   ```  

2. **Upload an image**:  
   - Supported formats: JPG, JPEG, PNG  

3. **Predict**:  
   - Click the "Predict" button to view the detected facial expression.  

## Dataset  

The model was trained on the **FER-2013 Dataset**, a publicly available dataset containing labeled facial expressions.  

**Dataset Link**: [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  

The model classifies images into the following categories:  
- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

## Model Details  

The model is a convolutional neural network built with TensorFlow. It was trained on 48x48 pixel grayscale images of facial expressions. Key steps include:  
- Data preprocessing and augmentation  
- Model training and evaluation  

## Deployment  

The application is deployed locally using **Streamlit**. A cloud-hosted version is also available.  

## Contributing  

Contributions are welcome! Feel free to fork the repository and submit pull requests with improvements.  

## License  

This project is licensed under the MIT License. See `LICENSE` for more details.  

## Contact  

For questions or suggestions, reach out to:  
- **Email**: ahmed.hamdii.kamal@gmail.com  
- **GitHub**: [hamdii10](https://github.com/hamdii10)  
