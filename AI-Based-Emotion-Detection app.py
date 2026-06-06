import streamlit as st
import cv2
import textwrap
import tempfile
import os
import moviepy.editor as mp
import librosa
import base64
import shutil
from fer import FER
from collections import Counter
from moviepy.editor import ImageSequenceClip
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from nltk.tokenize import sent_tokenize
import nltk

# Attempt to download punkt and punkt_tab if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


# Initialize the FER model for emotion detection
emotion_detector = FER(mtcnn=True)

# Load Hugging Face Whisper model for multilingual speech-to-text
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Load a sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to extract audio from video
def extract_audio(video_file):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        tmp_video.write(video_file.read())
        video = mp.VideoFileClip(tmp_video.name)

        # Check if the video has audio
        if video.audio is None:
            return tmp_video.name, None  # No audio track

        # Extract audio if available
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        return tmp_video.name, audio_path

# Transcribe audio using Whisper
def transcribe_audio(audio_file):
    audio, rate = librosa.load(audio_file, sr=16000)
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt").input_features
    predicted_ids = model.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Streamlit component to display live captions
def display_live_captions():
    placeholder = st.empty()  # Placeholder for live captions
    return placeholder

# Initialize a dictionary to store captions with their time ranges
captions_dict = {}

# Extract audio chunks with precise time intervals
def transcribe_audio_with_timestamps(audio_path, chunk_duration=5):
    audio, rate = librosa.load(audio_path, sr=16000)
    total_duration = librosa.get_duration(y=audio, sr=rate)
    captions_with_timestamps = []

    for start in range(0, int(total_duration), chunk_duration):
        end = min(start + chunk_duration, total_duration)
        chunk_audio = audio[int(start * rate):int(end * rate)]
        inputs = processor(chunk_audio, sampling_rate=rate, return_tensors="pt").input_features
        predicted_ids = model.generate(inputs)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        captions_with_timestamps.append({
            "start": start,
            "end": end,
            "text": transcription
        })
    return captions_with_timestamps


def overlay_bottom_caption(frame, text, max_width=40, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, thickness=1, color=(255, 255, 255)):
    # Wrap text to fit within the max_width
    wrapped_text = textwrap.wrap(text, width=max_width)

    # Determine frame dimensions
    frame_height, frame_width, _ = frame.shape
    text_height = 18  # Approximate height of each line of text

    # Calculate starting y-coordinate for bottom alignment with small margin
    total_text_height = len(wrapped_text) * (text_height + 5)
    y_start = frame_height - total_text_height - 15  # Margin from bottom edge

    for i, line in enumerate(wrapped_text):
        # Calculate text size and position
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_width, text_height = text_size
        x = 20  # Small margin from the left side
        y = y_start + i * (text_height + 15)

        # Draw a semi-transparent background for each line
        cv2.rectangle(frame,
                      (x - 10, y - text_height - 10),
                      (frame_width - 20, y + 10),  # Leave space on the right side
                      (0, 0, 0),
                      -1)  # Black semi-transparent background

        # Overlay the text on top, centered horizontally
        cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    return frame


# Perform sentence-wise sentiment analysis
def analyze_sentence_sentiment(transcript):
    sentences = sent_tokenize(transcript)
    sentence_sentiments = []
    for sentence in sentences:
        sentiment_score = sentiment_pipeline(sentence)[0]
        sentence_sentiments.append({"sentence": sentence, "sentiment": sentiment_score})
    return sentence_sentiments

# Calculate overall sentiment
def calculate_overall_sentiment(sentence_sentiments):
    sentiment_values = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
    total_score = sum(sentiment_values[sentiment['sentiment']['label']] for sentiment in sentence_sentiments)
    count = len(sentence_sentiments)
    avg_score = total_score / count if count > 0 else 0
    overall_sentiment = "POSITIVE" if avg_score > 0 else "NEGATIVE" if avg_score < 0 else "NEUTRAL"
    return overall_sentiment, avg_score

# Function to add background image
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: #F5F5F5;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Main Streamlit app function
def main():
    # Set up the page configuration
    st.set_page_config(page_title="EmoSentia: Real-time Emotion & Sentiment Detection", layout="wide")

    # Titles and Headers with custom CSS
    st.markdown(
        """
        <style>
        .title {
            font-size: 3em;
            font-weight: bold;
            color: #FFFFFF;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
            text-align: center;
            margin-top: 20px;
        }
        .subheader {
            font-size: 1.5em;
            color: #DCDCDC;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-text {
           color: #FFD700; /* Gold color for 'Upload a video' */
           font-size: 1.3em;
           font-weight: bold;
           text-align: center;
           margin-bottom: 10px;
        }
        .preview-text {
          color: #00CED1; /* Dark Turquoise for 'Video Preview' */
          font-size: 1.4em;
          font-weight: bold;
          text-align: center;
        }
        .speed-label {
          color: #FF4500; /* OrangeRed for 'Playback Speed' label */
          font-size: 1.2em;
          font-weight: bold;
          margin-top: 10px;
        }
        .speed-select {
           font-size: 1em;
           color: #333333;
           background-color: #FFFACD; /* LemonChiffon background for dropdown */
           padding: 5px;
           border-radius: 5px;
           border: 1px solid #FF4500;
         }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<p class="title">EmoSentia: Real-time Emotion & Sentiment Detection in Video Speech</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload your video and analyze emotions in video speech.</p>', unsafe_allow_html=True)
    # Background image
    add_bg_from_local('/content/sarang bg.jpg')  # Ensure the image is in the right path

    # Video Upload and Display with playback speed control
    st.markdown('<p class="upload-text">Upload a video file and preview it:</p>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
             tmp_path = tmp_file.name
             uploaded_video.seek(0)  # Reset file pointer to the start
             shutil.copyfileobj(uploaded_video, tmp_file)

        # Step 2: Read the saved video and encode it as base64 for HTML embedding
        with open(tmp_path, "rb") as f:
             video_bytes = f.read()
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")

        # HTML for video display with playback speed control
        video_html = f"""
        <div style="text-align: center;">
             <video id="videoPlayer" width="640" height="360" controls>
                   <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                   Your browser does not support the video tag.
             </video>
             <br>
             <label for="playbackSpeed" class="speed-label">Playback Speed:</label>
             <select id="playbackSpeed" class="speed-select" onchange="document.getElementById('videoPlayer').playbackRate = this.value">
                  <option value="0.5">0.5x</option>
                  <option value="1" selected>1x (Normal)</option>
                  <option value="1.5">1.5x</option>
                  <option value="2">2x</option>
            </select>
        </div>
        """
        st.components.v1.html(video_html, height=450)

        st.info("Extracting audio from video...")
        video_path, audio_path = extract_audio(open(tmp_path, "rb"))

        if audio_path is None:
             st.warning("The uploaded video has no audio. Only emotion analysis will be performed.")

        # Video and frame settings
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_skip = 5
        resize_factor = 0.5

        # Emotion processing variables
        processed_frames = []
        emotion_summary = Counter()
        frame_count = 0

        captions = transcribe_audio_with_timestamps(audio_path)

        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Calculate the time for the current frame
            frame_time = frame_count / fps
            current_caption = ""
            for caption in captions:
                if caption["start"] <= frame_time <= caption["end"]:
                     current_caption = caption["text"]
                     break

            # Overlay centered caption with wrapping
            if current_caption:
                frame = overlay_bottom_caption(frame, current_caption, max_width=40)

            if frame_count % frame_skip == 0:
                frame = frame.copy()  # Ensure frame is writable
                small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
                results = emotion_detector.detect_emotions(small_frame)

                for result in results:
                    bounding_box = [int(coord / resize_factor) for coord in result["box"]]
                    emotions = result["emotions"]
                    top_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[top_emotion]
                    emotion_summary[top_emotion] += 1

                    x, y, w, h = bounding_box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{top_emotion} ({confidence:.2f})"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=2)

                processed_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_count += 1

        cap.release()

        # Create video from processed frames
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        clip = ImageSequenceClip(processed_frames, fps=max(fps // frame_skip, 1))
        audio_clip = mp.AudioFileClip(audio_path) if audio_path else None
        clip = clip.set_audio(audio_clip) if audio_clip else clip
        clip.write_videofile(output_video_path, codec="libx264", bitrate="5000k", audio=True)


        # Display processed video
        st.write("### Processed Video with Emotion Detection")

        # Read the processed video file and encode it as base64
        with open(output_video_path, "rb") as f:
              processed_video_bytes = f.read()
        processed_video_base64 = base64.b64encode(processed_video_bytes).decode("utf-8")

        # HTML video player with the base64-encoded processed video
        processed_video_html = f"""
        <div style="text-align: center;">
                <video width="640" height="360" controls>
                <source src="data:video/mp4;base64,{processed_video_base64}" type="video/mp4">
                Your browser does not support the video tag.
        </video>
        </div>
        """

        # Display the processed video using Streamlit HTML component
        st.components.v1.html(processed_video_html, height=450)

        # Add download button for the processed video
        with open(output_video_path, "rb") as video_file:
            video_bytes = video_file.read()
            # Custom HTML and CSS for a styled download button
            download_button_html = f"""
            <div style="text-align: center; margin-top: 20px;">
                 <a href="data:video/mp4;base64,{base64.b64encode(video_bytes).decode('utf-8')}" download="emotion_detected_video.mp4" style="
                    display: inline-block;
                    font-size: 1.2em;
                    font-weight: bold;
                    color: green;
                    text-decoration: none;
                    padding: 10px 20px;
                    border: 2px solid green;
                    border-radius: 5px;
                    background-color: #F0FFF0;
                    transition: background-color 0.3s, color 0.3s;
                 " onmouseover="this.style.backgroundColor='#98FB98'; this.style.color='#006400';" onmouseout="this.style.backgroundColor='#F0FFF0'; this.style.color='green';">
                    Download Emotion Detected Video
                 </a>
            </div>
            """

        # Display the styled button using Streamlit's HTML component
        st.markdown(download_button_html, unsafe_allow_html=True)


        # Display dominant emotion summary
        if emotion_summary:
            dominant_emotion = emotion_summary.most_common(1)[0][0]
            st.write(f"### Dominant Emotion in Video: {dominant_emotion.capitalize()}")
            st.write("Emotion Breakdown:")
            for emotion, count in emotion_summary.items():
                st.write(f"{emotion.capitalize()}: {count} frames")

        if audio_path is None:
            st.warning("The uploaded video has no audio. Only emotion analysis was performed.")
        else:
            st.info("Transcribing audio...")
            transcript = transcribe_audio(audio_path)
            st.write("Transcript:", transcript)

            st.info("Performing sentence-wise sentiment analysis...")
            sentence_sentiments = analyze_sentence_sentiment(transcript)
            for sentiment in sentence_sentiments:
                sentence, score = sentiment["sentence"], sentiment["sentiment"]
                label = score['label']
                st.write(f"Sentence: {sentence}")
                st.write(f"Sentiment: {label} (Confidence: {score['score']:.2f})")
                st.write("---")

            # Display overall sentiment
            overall_sentiment, avg_score = calculate_overall_sentiment(sentence_sentiments)
            st.write(f"**Overall Sentiment:** {overall_sentiment} (Average Score: {avg_score:.2f})")

        # Clean up temporary files
        os.remove(video_path)
        if audio_path:
            os.remove(audio_path)
        os.remove(output_video_path)

if __name__ == "__main__":
    main()
