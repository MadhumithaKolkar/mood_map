// Access the webcam
const webcam = document.getElementById('webcam');
let videoStream = null;

// Start the webcam feed
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        webcam.srcObject = stream;
        videoStream = stream;
    })
    .catch((error) => {
        console.error("Webcam error:", error);
        alert("Unable to access the webcam. Please check your camera settings.");
    });

// Function to stop the webcam
function stopWebcam() {
    if (videoStream) {
        const tracks = videoStream.getTracks();
        tracks.forEach((track) => track.stop());
    }
}

// Global variables for audio recording
let mediaRecorder = null;
let audioChunks = [];

// Function to start audio recording
function startAudioRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then((stream) => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                audioChunks = [];
                processAudio(audioBlob);
            };
        })
        .catch((error) => {
            console.error("Audio recording error:", error);
            alert("Unable to access the microphone. Please check your microphone settings.");
        });
}

// Function to process audio
function processAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    fetch('/api/process-audio', {
        method: 'POST',
        body: formData
    })
        .then((response) => response.json())
        .then((data) => {
            console.log("Audio response:", data);
            updateChatResponse(data.chat_response);
        })
        .catch((error) => {
            console.error("Error processing audio:", error);
        });
}

// Function to send video emotion data
function sendVideoEmotionData(emotion) {
    fetch('/api/process-video', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ emotion })
    })
        .then((response) => response.json())
        .then((data) => {
            console.log("Video response:", data);
            updateChatResponse(data.chat_response);
        })
        .catch((error) => {
            console.error("Error processing video emotion:", error);
        });
}

// Function to update the ChatGPT response in the UI
function updateChatResponse(response) {
    const chatResponseElement = document.getElementById('chat-response');
    chatResponseElement.textContent = response;
}

// Button event listeners
document.getElementById('speak-btn').addEventListener('click', () => {
    console.log("Speak button clicked");
    startAudioRecording();
    // Logic to start video-based emotion recognition
    startVideoEmotionRecognition();
});

document.getElementById('send-btn').addEventListener('click', () => {
    console.log("Send button clicked");
    if (mediaRecorder) {
        mediaRecorder.stop(); // Stop audio recording
    }
    stopVideoEmotionRecognition(); // Stop video emotion detection
    stopWebcam();
});

// Function to start video emotion recognition
function startVideoEmotionRecognition() {
    // Logic to process video frames for emotion detection
    setInterval(() => {
        const videoCanvas = document.createElement('canvas');
        videoCanvas.width = webcam.videoWidth;
        videoCanvas.height = webcam.videoHeight;
        const ctx = videoCanvas.getContext('2d');
        ctx.drawImage(webcam, 0, 0, videoCanvas.width, videoCanvas.height);

        videoCanvas.toBlob((blob) => {
            const formData = new FormData();
            formData.append('frame', blob, 'frame.jpg');

            fetch('/api/process-frame', {
                method: 'POST',
                body: formData
            })
                .then((response) => response.json())
                .then((data) => {
                    console.log("Video emotion:", data);
                    sendVideoEmotionData(data.emotion);
                })
                .catch((error) => {
                    console.error("Error processing frame:", error);
                });
        }, 'image/jpeg');
    }, 1000); // Process a frame every second
}

// Function to stop video emotion recognition
function stopVideoEmotionRecognition() {
    clearInterval();
}