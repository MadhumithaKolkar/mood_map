# 🎭 MoodMap: Giving LLMs the Power to See, Hear, and Feel Emotion

**MoodMap** is a **multimodal AI system** that gives large language models (LLMs) the ability to **see, hear, and feel human emotion**. By combining facial expression recognition, voice tone analysis, and natural language understanding, MoodMap creates a deeply empathetic experience—transforming static chatbots into emotionally-aware digital companions.

[![Watch the video](https://img.youtube.com/vi/HFfQ9Y2GsL4/maxresdefault.jpg)](https://youtu.be/HFfQ9Y2GsL4)

It senses how you're feeling in the moment, processes both your **voice and facial cues**, and responds with human-like empathy through a powerful LLM and realistic text-to-speech.

## 🧠 Features
- 🎤 **Speech Emotion Recognition** (custom-trained model)
- 🎥 **Facial Emotion Detection** using DeepFace
- 💬 **Speech Transcription** with SpeechRecognition
- 🤖 **Context-Aware GPT-4 Responses** (OpenAI)
- 🗣️ **Human-Like Voice Reply** (Text-to-Speech)
- 🖥️ Real-time webcam feed with live emotion overlay
- 🔁 **Fully Multithreaded** & **Asynchronous** processing

## 📦 Project Structure
```
project/
├── audio_data/
│   └── recorded_data/
│       └── recorded_audio.wav
├── last_50.keras                # Trained speech emotion model
├── last_scaler.pkl              # Feature scaler for audio features
├── main.py                      # Main execution script
├── requirements.txt
└── README.md
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/moodmap.git
cd moodmap
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> Make sure you have FFmpeg installed and added to your system path for audio processing.

## 🔑 API Key
Set your OpenAI API key as an environment variable:
```bash
export OPEN_AI_API_KEY=your_openai_key_here
```

Or update the code to fetch from `os.getenv("OPEN_AI_API_KEY")`.

## 🚀 Running the App
```bash
python main.py
```

- Press `Q` to quit the webcam feed.
- The system records your voice, detects emotion from both video and audio, and responds with a personalized message and voice.

## 🎓 Tech Stack
- **OpenCV** – Real-time webcam and overlay
- **DeepFace** – Facial emotion recognition
- **TensorFlow / Keras** – Custom LSTM model for speech emotion
- **Librosa** – Audio feature extraction
- **SpeechRecognition** – Voice-to-text conversion
- **OpenAI GPT-4** – Chatbot intelligence
- **pyttsx3** – Offline text-to-speech
- **Threading** – For asynchronous video/audio/chat processes

## 🗂 Emotion Labels
Speech emotion model recognizes the following emotions:
```
['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
```

## 🤖 Model Training
The speech emotion model was trained from scratch using combined datasets:
- RAVDESS
- CREMA-D
- TESS
- Custom dataset

Audio features used:
- Zero Crossing Rate
- Root Mean Square Energy
- 13 MFCCs
- Concatenated and padded to uniform length (1620 features)

## ⚠️ Known Issues
- Facial detection requires adequate lighting
- Voice transcription can misfire in noisy environments
- Response time may vary slightly depending on system resources

## 🙌 Acknowledgments
- [DeepFace](https://github.com/serengil/deepface)
- [Librosa](https://librosa.org/)
- [OpenAI GPT](https://platform.openai.com/)
- Emotion datasets: [RAVDESS], [CREMA-D], [TESS]

## 📜 License
MIT License

## Created by Madhumitha Kolkar 2025
