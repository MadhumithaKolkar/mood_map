# Created by Madhumitha Kolkar 2025

import threading
import queue
import cv2
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from joblib import load
import os
import sounddevice as sd
from scipy.io.wavfile import write
import simpleaudio as sa
from deepface import DeepFace
import speech_recognition as sr
import openai  # Importing OpenAI library
import pyttsx3
from playsound import playsound
import os

# Set your OpenAI API key
openai.api_key = OPEN_AI_API_KEY  # Replace with your actual API key, use os.getenv() preferrably 

# Emotion labels
emotion_dict = {
    0: 'angry',
    1: 'calm',
    2: 'disgust',
    3: 'fear',
    4: 'happy',
    5: 'neutral',
    6: 'sad',
    7: 'surprise'
}

# Load pre-trained scaler and model
scaler = load('last_scaler.pkl')  # Replace with your scaler file
model = load_model('last_50.keras')  # Replace with your model file

# Thread-safe queue for video frames
frame_queue = queue.Queue()

# Stop event to signal threads to terminate
stop_event = threading.Event()

import pyttsx3

def speak_text(text):
    """
    Converts text to speech with a balanced American female voice and plays it out loud.
    """
    try:
        # Initialize the text-to-speech engine
        engine = pyttsx3.init()

        # Get the available voices
        voices = engine.getProperty('voices')

        # Set to an American female voice
        for voice in voices:
            if "female" in voice.name.lower() and "english" in voice.languages[0].decode("utf-8").lower():
                engine.setProperty('voice', voice.id)
                break
        else:
            print("[TTS] American female voice not found, using default voice.")

        # Adjust the rate for normal tone
        engine.setProperty('rate', 180)  # Default is ~200; reduce for a slower tone

        # Adjust volume
        engine.setProperty('volume', 0.9)  # Set to 90% volume for clarity

        # Speak the text
        engine.say(text)
        engine.runAndWait()
        print("[TTS] Playing response out loud with American female voice.")
    except Exception as e:
        print(f"[TTS] Error: {e}")


# ChatGPT API interaction
def get_chatgpt_response(emotion_summary):
    """
    Sends combined emotion and transcription data to ChatGPT and retrieves the response.
    """
    try:
        print("[CHATGPT] Sending data to ChatGPT...")
        messages = [
            {"role": "system", "content": "You are a helpful and empathetic assistant."},
            {
                "role": "user",
                "content": f"""
                I have the following emotion data and transcription:
                {emotion_summary}
                Based on this information, provide an insightful and empathetic response and also show interest in the user.
                If the user is sad or happy then acknowledge it as - "Hey you look quite sad , do you wanna talk about it ?" or "Hey you look like you're in a good mood"
                 or whatever the emotion is. You must sound human like and like a friend , be casual.
                """
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )

        reply = response["choices"][0]["message"]["content"]
        print(f"[CHATGPT] Response: {reply}")
        speak_text(reply)
        return reply

    except Exception as e:
        print(f"[CHATGPT] Error: {e}")
        return "Error communicating with ChatGPT."


# Audio processing functions
def extract_features(data, sr, frame_length=2048, hop_length=512):
    stft_data = np.abs(librosa.stft(data, n_fft=frame_length, hop_length=hop_length))
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length).squeeze()
    rmse = librosa.feature.rms(S=stft_data).squeeze()
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), sr=sr, n_mfcc=13).T.ravel()
    return np.hstack([zcr, rmse, mfcc])


def get_features(path, duration=2.5, offset=0.6, sr=22050, frame_length=2048, hop_length=512):
    data, sr = librosa.load(path, duration=duration, offset=offset)
    features = extract_features(data, sr, frame_length, hop_length)
    target_length = 1620
    if len(features) < target_length:
        features = np.pad(features, (0, target_length - len(features)), mode='constant')
    elif len(features) > target_length:
        features = features[:target_length]
    return features


def predict_emotion_from_file(file_path):
    features = get_features(file_path)
    features = features.reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_emotion = emotion_dict[predicted_class]
    print(f"[AUDIO] File: {file_path}, Predicted Emotion: {predicted_emotion}")
    return predicted_emotion


def play_audio(file_path):
    wave_obj = sa.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()


def record_audio(duration=5, fs=22050):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return recording.flatten()


def save_recorded_audio(recording, filename="./audio_data/recorded_data/recorded_audio.wav", fs=22050):
    write(filename, fs, recording)
    print(f"[AUDIO] Audio saved as {filename}")


def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        print("[AUDIO] Transcribing audio...")
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"[AUDIO] Transcription: {text}")
            return text
        except sr.UnknownValueError:
            print("[AUDIO] Could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"[AUDIO] Error with the Speech Recognition service: {e}")
            return None


# Video processing functions
def detect_emotion_from_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
        except Exception as e:
            emotion = "Error"
            print(f"[VIDEO] Emotion Detection Error: {e}")

        cv2.putText(frame, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        frame_queue.put((frame, emotion))

    cap.release()


def display_frames():
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame, _ = frame_queue.get()
            cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()


# Audio and Video integration
def record_and_predict(duration=5):
    while not stop_event.is_set():
        # Audio emotion and transcription
        recorded_audio = record_audio(duration)
        audio_path = './audio_data/recorded_data/recorded_audio.wav'
        save_recorded_audio(recorded_audio, filename=audio_path)
        audio_emotion = predict_emotion_from_file(audio_path)
        transcription = transcribe_audio(audio_path)

        # Video emotion
        video_emotion = None
        if not frame_queue.empty():
            _, video_emotion = frame_queue.get()

        # Prepare data for ChatGPT
        emotion_summary = f"""
        Audio Emotion: {audio_emotion}
        Video Emotion: {video_emotion}
        Transcription: {transcription}
        """
        if transcription:
            chatgpt_response = get_chatgpt_response(emotion_summary)
            print(f"[FINAL RESPONSE] {chatgpt_response}")


# Main function
def main():
    audio_thread = threading.Thread(target=record_and_predict, args=(5,))
    video_thread = threading.Thread(target=detect_emotion_from_video)

    audio_thread.start()
    video_thread.start()

    display_frames()

    audio_thread.join()
    video_thread.join()


if __name__ == "__main__":
    main()
