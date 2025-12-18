import sys
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from faster_whisper import WhisperModel


# --------------------------
# Whisper 모델 로드 (GPU)
# --------------------------
model = WhisperModel("small", device="cuda", compute_type="float16")


# --------------------------
# WAV 파일 저장
# --------------------------
def record_audio(duration=4, sample_rate=16000, filename="voice_input.wav"):
    print("🎤 녹음 시작...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, sample_rate, audio)
    print("🎤 녹음 완료:", filename)
    return filename


# --------------------------
# Whisper로 한국어 텍스트 변환
# --------------------------
def transcribe_korean(audio_path):
    print("🔍 음성 인식 중...")
    segments, info = model.transcribe(audio_path, language="ko")
    text = "".join([seg.text for seg in segments])
    print("📝 인식 결과:", text)
    return text


# --------------------------
# PySide6 UI
# --------------------------
class VoiceKioskUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("음성 인식 키오스크 테스트")
        self.setFixedSize(450, 350)

        layout = QVBoxLayout()

        self.label = QLabel("🎤 '음성 입력' 버튼을 눌러주세요.")
        layout.addWidget(self.label)

        self.btn_record = QPushButton("🎙 음성 입력 시작 (4초)")
        self.btn_record.clicked.connect(self.record_and_recognize)
        layout.addWidget(self.btn_record)

        self.textbox = QTextEdit()
        layout.addWidget(self.textbox)

        self.setLayout(layout)

    def record_and_recognize(self):
        audio_path = record_audio()

        text = transcribe_korean(audio_path)

        self.textbox.setText(text)


# --------------------------
# 실행
# --------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = VoiceKioskUI()
    ui.show()
    sys.exit(app.exec())
