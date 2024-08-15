import cv2
import speech_recognition as sr
from pydub import AudioSegment
from deepface import DeepFace
from collections import defaultdict
import os
from pyAudioAnalysis import audioTrainTest as aT
from janome.tokenizer import Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tempfile

# 感情のラベルを日本語に変換する辞書
emotion_labels = {
    'angry': '怒り',
    'disgust': '嫌悪',
    'fear': '恐怖',
    'happy': '幸せ',
    'sad': '悲しみ',
    'surprise': '驚き',
    'neutral': '中立'
}

def classify_emotion(audio_path):
    model_path = os.path.join(os.getcwd(), "models/svm_rbf_4class")
    result, probability, _ = aT.file_classification(audio_path, model_path, "svm")
    emotions = ['angry', 'disgust', 'fear', 'happy']
    emotion = emotions[int(result)]
    return emotion

def analyze_sentiment_japanese(text):
    # 日本語のテキストを形態素解析し、VADERで感情分析
    tokenizer = Tokenizer()
    words = " ".join([token.surface for token in tokenizer.tokenize(text)])
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(words)
    
    # 感情分析の結果を返す
    sentiment_result = f"ポジティブ: {sentiment['pos']}, ニュートラル: {sentiment['neu']}, ネガティブ: {sentiment['neg']}"
    return sentiment_result

def analyze_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    audio_path = "extracted_audio.wav"
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='ja-JP')
            audio_text_summary = f"音声のテキスト化結果:\n{text}\n"
        except sr.UnknownValueError:
            audio_text_summary = "音声を理解できませんでした。\n"
            text = ""
        except sr.RequestError as e:
            audio_text_summary = f"音声認識サービスへのリクエストでエラーが発生しました: {e}\n"
            text = ""

    if text:
        text_sentiment_summary = analyze_sentiment_japanese(text)
    else:
        text_sentiment_summary = "テキスト内容が取得できなかったため、感情分析を行えませんでした。\n"

    voice_emotion = classify_emotion(audio_path)
    voice_emotion_summary = f"声色からの感情分析結果: {emotion_labels.get(voice_emotion, voice_emotion)}\n"

    cap = cv2.VideoCapture(video_path)
    frame_interval = 5
    frame_number = 0
    emotion_summary = defaultdict(float)
    total_frames_analyzed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % frame_interval == 0:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                for res in result:
                    for emotion, score in res['emotion'].items():
                        emotion_summary[emotion] += score
                    total_frames_analyzed += 1
            except Exception as e:
                st.write(f"フレーム {frame_number} でエラーが発生しました: {e}")
        frame_number += 1
    cap.release()

    emotion_summary_text = "動画の表情からの感情分析結果の要約:\n"
    for emotion, total_score in emotion_summary.items():
        average_score = total_score / total_frames_analyzed
        emotion_japanese = emotion_labels.get(emotion, emotion)
        emotion_summary_text += f"{emotion_japanese}: 平均 {average_score:.2f}%\n"

    dominant_emotion = max(emotion_summary, key=emotion_summary.get)
    emotion_summary_text += f"\n最も強い感情: {emotion_labels.get(dominant_emotion, dominant_emotion)}\n"

    return audio_text_summary, text_sentiment_summary, voice_emotion_summary, emotion_summary_text

# Streamlit UIの構築
st.title("動画からの感情分析アプリ")

uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.write("動画を分析しています...")

    audio_text_summary, text_sentiment_summary, voice_emotion_summary, emotion_summary_text = analyze_video(uploaded_file)

    st.subheader("=== 音声のテキスト化結果 ===")
    st.text(audio_text_summary)

    st.subheader("=== テキスト内容からの感情分析結果 ===")
    st.text(text_sentiment_summary)

    st.subheader("=== 声色からの感情分析結果 ===")
    st.text(voice_emotion_summary)

    st.subheader("=== 動画の表情からの感情分析結果 ===")
    st.text(emotion_summary_text)
