# Vision Companion AI

A sophisticated, vision-enabled AI assistant that combines real-time conversation, object detection, and an interactive 3D avatar. This project uses Google Gemini for conversational AI, ElevenLabs for realistic text-to-speech, and YOLOv8 for live object detection from a webcam feed.

## Features

-   **Conversational AI:** Engage in natural, real-time conversations powered by Google Gemini.
-   **Realistic Voice:** Hear the AI's responses in a high-quality voice from ElevenLabs.
-   **Live Vision System:** The AI can "see" and describe objects in real-time using a webcam and the YOLOv8 object detection model.
-   **Interactive 3D Avatar:** A customizable VRM-based 3D avatar that animates, expresses emotion, and reacts during conversation.
-   **Speech Recognition:** Talk to the AI using your microphone.
-   **Dynamic UI:** An interactive interface showing the avatar, controls, and vision analysis.

## Requirements

Make sure you have Python 3.8+ installed. You can install the necessary packages using pip:

```bash
pip install google-generativeai elevenlabs pyaudio SpeechRecognition opencv-python pillow pygame numpy onnxruntime
```

You will also need:
-   A webcam connected to your computer.
-   The `yolov8n.onnx` model file in the same directory as the script.

## Configuration

Before running the application, you need to configure your API keys and model paths in the `characters.py` file.

1.  **API Keys:**
    Open `characters.py` and replace the placeholder values for `GEMINI_API_KEY` and `ELEVENLABS_API_KEY` with your actual keys.

    ```python
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
    ELEVENLABS_API_KEY = "YOUR_ELEVENLABS_API_KEY"
    ```

    **For better security, it is highly recommended to use environment variables to store your API keys.**

2.  **VRM Model (Optional):**
    To use your own VRM avatar, update the `DEFAULT_VRM_PATH` to point to your `.vrm` file. If no file is found, a default preview avatar will be used.

    ```python
    DEFAULT_VRM_PATH = r"C:\path\to\your\avatar.vrm"
    ```

## Usage

1.  Navigate to the project directory in your terminal.
2.  Run the script:

    ```bash
    python characters.py
    ```

3.  Use the following keyboard controls in the application window:
    -   **`SPACE`**: Press and hold to talk to the AI.
    -   **`V`**: Toggle the live YOLOv8 vision feed window.
    -   **`U`**: Open a file dialog to upload a new VRM avatar.
    -   **Arrow Keys**: Rotate the 3D avatar.
    -   **`+/-`**: Zoom in and out.
    -   **`ESC`**: Exit the application.

