# Hybrid DNN–LSTM Based Indian Sign Language Recognition and Speech Synthesis System

## Project Overview

This project is a real-time Indian Sign Language (ISL) recognition and speech synthesis system that recognizes static alphabet gestures and custom dynamic gestures using MediaPipe hand landmarks and a Hybrid Deep Neural Network (DNN)–Long Short-Term Memory (LSTM) architecture.

The recognized hand gestures are converted into meaningful text and synthesized into speech, helping improve communication between people who use Indian Sign Language and those who may not understand it.

## Key Features

- Real-time hand gesture detection using a webcam.
- Static Indian Sign Language alphabet recognition.
- Dynamic gesture recognition using a Hybrid DNN–LSTM model.
- Hand landmark detection using MediaPipe Hands.
- Gesture-to-text conversion.
- Automatic sentence formation from recognized gestures.
- Paragraph formation support.
- Text-to-speech conversion using Google Text-to-Speech (gTTS).
- Confidence score display for predictions.
- Gesture prediction after holding the same gesture for 1.5 seconds.
- Real-time visual feedback and audio output.

## Technologies Used

- Python 3.10+
- TensorFlow / Keras
- MediaPipe Hands
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- gTTS
- Pygame
- Pillow
- Joblib

## Models and Architecture

### 1. MediaPipe Hands
Detects and extracts 21 hand landmarks from webcam frames.

### 2. Deep Neural Network (DNN)
Used for recognizing static alphabet gestures.

### 3. Long Short-Term Memory (LSTM)
Used for recognizing dynamic gestures by learning temporal patterns across sequences of hand movements.

### 4. Hybrid DNN–LSTM
Combines static and sequential gesture recognition to support different types of Indian Sign Language gestures.

### 5. Google Text-to-Speech
Converts recognized text into spoken audio.

## Software Requirements

- Python 3.10 or later
- Webcam
- Windows or macOS
- Working internet connection for online gTTS speech generation

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ISL-to-Text-to-Speech-Recognition
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the real-time recognition application:

```bash
python dual_mode_isl_test.py
```

Allow webcam access when prompted.

## Keyboard Controls

| Key | Function |
|---|---|
| A | Switch to Alphabet Recognition Mode |
| G | Switch to Gesture Recognition Mode |
| Q | Quit the application |
| Spacebar | Form sentences using recognized gestures |
| P | Form paragraphs |
| V | Generate voice output |

## Application Output

The application displays:

- Detected hand landmarks.
- Predicted alphabet or gesture.
- Prediction confidence score.
- Recognized text.
- Sentence and paragraph formation.
- Synthesized speech output.

The system uses a 1.5-second gesture holding mechanism to reduce accidental predictions.

## Project Objective

The main objective of this project is to develop an intelligent and accessible communication system that converts Indian Sign Language gestures into text and speech using computer vision and deep learning techniques.

## Future Enhancements

- Expand the vocabulary of recognized gestures.
- Improve recognition accuracy under different lighting conditions.
- Support continuous sign language sentence recognition.
- Develop a mobile application.
- Improve speech generation and multilingual support.

## Author

Jwala Sainag Mahadev

## Acknowledgement

This project demonstrates the application of computer vision, deep learning, sequence modeling, and speech synthesis for accessible communication.

### Screenshots
<img width="712" height="630" alt="image" src="https://github.com/user-attachments/assets/91700b32-56eb-43b9-9a71-0690063d2cfc" />
<img width="710" height="467" alt="image" src="https://github.com/user-attachments/assets/1baa80cf-8f3e-446a-97b3-f5836baeced1" />
<img width="787" height="447" alt="image" src="https://github.com/user-attachments/assets/ea28cd4a-6cf8-48f4-861f-76264818e61e" />
<img width="783" height="442" alt="image" src="https://github.com/user-attachments/assets/34cf0d30-db8a-4fad-bf54-b573eaff71d5" />
<img width="788" height="445" alt="image" src="https://github.com/user-attachments/assets/1f0f2200-4fcb-4f5d-b570-23bcce05c672" />

