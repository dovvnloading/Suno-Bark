import sys
import nltk
import traceback
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QWidget, QTextEdit, QPushButton, QLabel, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QLineEdit, QMessageBox, QComboBox, QProgressBar, QSpacerItem,
                             QSizePolicy, QGroupBox, QRadioButton, QButtonGroup, QSlider)
from PySide6.QtCore import QMutex, QMutexLocker, Qt, QTimer, QThread, Signal, QPoint, QSize
from PySide6.QtGui import QFont, QPainter, QColor, QPalette, QPen, QIcon
from transformers import AutoModelForTextToWaveform, AutoProcessor
import torch
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import io
import os
try:
    import sounddevice as sd
except ImportError:
    sd = None

# Ensure the 'punkt' tokenizer is available
nltk.download('punkt', quiet=True)

# Helper function to find resource files
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class CustomTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_widget = parent
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.Highlight)

        self.setFixedHeight(35)
        self.setObjectName("customTitleBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(10)

        self.title_label = QLabel(self.parent_widget.windowTitle(), self)
        self.title_label.setFont(QFont("Segoe UI", 10))
        self.title_label.setStyleSheet("color: #CCCCCC;")

        self.minimize_button = QPushButton("—", self)
        self.maximize_button = QPushButton("⬜", self)
        self.close_button = QPushButton("✕", self)

        for btn, obj_name in [(self.minimize_button, "minimizeButton"),
                              (self.maximize_button, "maximizeButton"),
                              (self.close_button, "closeButton")]:
            btn.setFixedSize(40, 30)
            btn.setObjectName(obj_name)

        layout.addWidget(self.title_label)
        layout.addStretch()
        layout.addWidget(self.minimize_button)
        layout.addWidget(self.maximize_button)
        layout.addWidget(self.close_button)

        self.minimize_button.clicked.connect(self.parent_widget.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximize)
        self.close_button.clicked.connect(self.parent_widget.close)

        self.start_pos = None
        self.is_maximized = False

    def toggle_maximize(self):
        if self.is_maximized:
            self.parent_widget.showNormal()
            self.is_maximized = False
            self.maximize_button.setText("⬜")
        else:
            self.parent_widget.showMaximized()
            self.is_maximized = True
            self.maximize_button.setText("❐")
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPosition().toPoint() - self.parent_widget.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.start_pos is not None and event.buttons() == Qt.LeftButton:
            if self.is_maximized:
                self.toggle_maximize()
                new_pos = event.globalPosition().toPoint()
                half_width = self.parent_widget.width() // 2
                self.start_pos = QPoint(half_width, self.start_pos.y())
                self.parent_widget.move(new_pos - self.start_pos)
            else:
                self.parent_widget.move(event.globalPosition().toPoint() - self.start_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.start_pos = None
        event.accept()

class LoadingWheel(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 0)
        self.setTextVisible(False)
        self.setFixedSize(30, 30)
        self.hide()

class AudioGenerationThread(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, processor, model, text, voice_preset, sample_rate, device):
        super().__init__()
        self.processor = processor
        self.model = model
        self.text = text
        self.voice_preset = voice_preset
        self.sample_rate = sample_rate
        self.device = device

    def run(self):
        try:
            torch.manual_seed(0)
            inputs = self.processor(self.text, voice_preset=self.voice_preset, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                audio_data = self.model.generate(**inputs)

            self.finished.emit(audio_data)
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

class AudioPlaybackThread(QThread):
    finished = Signal()
    error = Signal(str)

    def __init__(self, audio_data_np, sample_rate):
        super().__init__()
        self.audio_data_np = audio_data_np
        self.sample_rate = sample_rate
        self._is_running = True

    def run(self):
        if not sd:
            self.error.emit("sounddevice library not found. Cannot play audio.")
            return

        try:
            sd.play(self.audio_data_np, self.sample_rate)
            sd.wait()
            
            if self._is_running:
                self.finished.emit()

        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def stop(self):
        self._is_running = False
        if sd:
            sd.stop()

class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.waveform = None
        self.total_duration = 0
        self.background_color = QColor(40, 40, 40)
        self.waveform_color = QColor(74, 158, 255)
        self.playback_line_color = QColor(255, 80, 80)
        self.playback_position = 0

    def plot_waveform(self, audio_data, sample_rate):
        if isinstance(audio_data, AudioSegment):
            samples = np.array(audio_data.get_array_of_samples())
            self.total_duration = len(audio_data)
        else:
            samples = audio_data.squeeze().cpu().numpy()
            self.total_duration = (len(samples) / sample_rate) * 1000
        
        if samples.size == 0:
            self.waveform = None
            self.update()
            return
            
        samples = samples / np.max(np.abs(samples))
        
        num_pixels = self.width() if self.width() > 0 else 1000
        resampled = np.interp(np.linspace(0, len(samples) - 1, num_pixels), np.arange(len(samples)), samples)
        
        self.waveform = resampled
        self.playback_position = 0
        self.update()

    def update_playback_position(self, position_ms):
        self.playback_position = position_ms
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.background_color)

        if self.waveform is None or self.total_duration <= 0:
            return

        pen = QPen(self.waveform_color)
        pen.setWidth(2)
        painter.setPen(pen)

        middle = self.height() / 2
        for x, sample in enumerate(self.waveform):
            y = int(middle * (1 - sample))
            painter.drawLine(x, int(middle), x, y)
        
        if self.playback_position > 0:
            pos_x = int((self.playback_position / self.total_duration) * self.width())
            painter.setPen(QPen(self.playback_line_color, 2))
            painter.drawLine(pos_x, 0, pos_x, self.height())
            
    def resizeEvent(self, event):
        super().resizeEvent(event)

class TextToSpeechApp(QWidget):
    update_waveform_signal = Signal(int)
    update_status_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self._thread_lock = QMutex()
    
        self.active_threads = []
        self.audio_data = None
        self.sample_rate = 24000
        self.audio_thread = None
        self.playback_thread = None
        
        self.current_position_ms = 0
        self.playback_start_timestamp = 0
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_playback_progress)

        self.setWindowTitle("AI Text-to-Speech")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setMinimumSize(800, 600)
        
        try:
            self.init_models()
            self.initUI()
            self.setNightModeStyle()
        
            self.update_waveform_signal.connect(self.waveform_widget.update_playback_position)
            self.update_status_signal.connect(self.status_label.setText)
        
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize application: {str(e)}")
            raise

    def init_models(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_name = "suno/bark"
        
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForTextToWaveform.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
        
            if torch.cuda.is_available():
                self.model.enable_cpu_offload()
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
            
    def closeEvent(self, event):
        with QMutexLocker(self._thread_lock):
            for thread in list(self.active_threads):
                if thread and thread.isRunning():
                    if hasattr(thread, 'stop'):
                        thread.stop()
                    thread.terminate()
                    thread.wait(500)
            self.active_threads.clear()
        
        if hasattr(self, 'model'):
            try:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error cleaning up model: {e}")
        
        event.accept()

    def generate_audio(self):
        try:
            if not self.text_edit.toPlainText().strip():
                raise ValueError("No text provided")
                
            if self.audio_thread and self.audio_thread.isRunning():
                self.audio_thread.stop()
                self.audio_thread.wait()
            
            self.loading_wheel.show()
            self.generate_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.status_label.setText("Generating audio...")
            
            input_text = self.preprocess_text(self.text_edit.toPlainText())
            selected_voice = self.voice_preset_combo.currentText()
            
            try:
                sample_rate = int(self.sample_rate_input.text()) if self.sample_rate_input.text() else 24000
                if sample_rate <= 0: raise ValueError()
                self.sample_rate = sample_rate
            except ValueError:
                self.status_label.setText("Invalid sample rate. Using default 24000.")
                self.sample_rate = 24000
            
            self.audio_thread = AudioGenerationThread(
                self.processor, self.model, input_text, selected_voice, 
                self.sample_rate, self.device
            )
            self.audio_thread.finished.connect(self.on_audio_generated)
            self.audio_thread.error.connect(self.on_audio_generation_error)
            self.audio_thread.finished.connect(self.on_thread_finished)
            
            with QMutexLocker(self._thread_lock):
                self.active_threads.append(self.audio_thread)
                self.audio_thread.start()
                
        except Exception as e:
            self.handle_error(f"Failed to start audio generation: {str(e)}\n\n{traceback.format_exc()}")

    def handle_error(self, message):
        self.status_label.setText(f"Error: Check console for details.")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"--- ERROR at {timestamp} ---\n{message}\n\n"
        self.loading_wheel.hide()
        self.generate_button.setEnabled(True)
        self.play_button.setEnabled(self.audio_data is not None)
        print(log_message)

    def preprocess_text(self, text):
        replacements = {
            "[laughter]": "♪ haha ♪", "[laughs]": "♪ haha ♪", "[sighs]": "♪ sigh ♪",
            "[music]": "♪ la la la ♪", "[gasps]": "♪ gasp ♪", "[clears throat]": "♪ ahem ♪"
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def on_audio_generated(self, audio_data):
        self.audio_data = audio_data
        self.on_generation_complete()

    def on_thread_finished(self):
        thread_to_remove = self.sender()
        if not thread_to_remove:
            return

        with QMutexLocker(self._thread_lock):
            if thread_to_remove in self.active_threads:
                self.active_threads.remove(thread_to_remove)
            
            if thread_to_remove is self.audio_thread:
                self.audio_thread = None
            if thread_to_remove is self.playback_thread:
                self.playback_thread = None

            thread_to_remove.deleteLater()

    def on_generation_complete(self):
        try:
            self.waveform_widget.plot_waveform(self.audio_data, self.sample_rate)
            duration_ms = self.waveform_widget.total_duration
            self.seek_slider.setRange(0, int(duration_ms))
            self.seek_slider.setEnabled(True)
            self.update_ui_after_generation()
        except Exception as e:
            self.handle_error(f"Error processing generated audio: {str(e)}\n\n{traceback.format_exc()}")

    def update_ui_after_generation(self):
        self.save_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.status_label.setText("Audio generated successfully.")
        self.loading_wheel.hide()
        self.generate_button.setEnabled(True)
        
    def play_audio(self):
        try:
            if self.playback_thread and self.playback_thread.isRunning():
                self.playback_thread.stop()
                return

            if self.audio_data is None:
                raise ValueError("No audio to play")
            
            if isinstance(self.audio_data, AudioSegment):
                samples = np.array(self.audio_data.get_array_of_samples()).astype(np.float32)
                samples /= 32768.0 
            else:
                samples = self.audio_data.squeeze().cpu().numpy().astype(np.float32)

            start_sample = int((self.current_position_ms / 1000) * self.sample_rate)
            playback_data = samples[start_sample:]

            self.playback_thread = AudioPlaybackThread(playback_data, self.sample_rate)
            self.playback_thread.finished.connect(self.on_playback_finished)
            self.playback_thread.finished.connect(self.on_thread_finished)
            self.playback_thread.error.connect(self.on_audio_generation_error)
            
            with QMutexLocker(self._thread_lock):
                self.active_threads.append(self.playback_thread)
                self.playback_thread.start()
            
            self.playback_start_timestamp = QTimer.singleShot(0, lambda: None)
            self.playback_timer.start(50) 
            self.play_button.setText("Stop Audio")
            self.update_status_signal.emit("Playing audio...")
            
        except Exception as e:
            self.handle_error(f"Error playing audio: {str(e)}\n\n{traceback.format_exc()}")
            
    def on_playback_finished(self):
        self.playback_timer.stop()
        self.play_button.setText("Play Audio")
        self.status_label.setText("Audio playback finished.")
        self.current_position_ms = 0
        self.seek_slider.setValue(0)
        self.update_waveform_signal.emit(0)

    def update_playback_progress(self):
        total_duration = self.waveform_widget.total_duration
        if total_duration <= 0 or not self.playback_thread or not self.playback_thread.isRunning():
            return

        self.current_position_ms += 50
        if self.current_position_ms > total_duration:
            self.current_position_ms = total_duration

        self.seek_slider.setValue(self.current_position_ms)
        self.update_waveform_signal.emit(self.current_position_ms)

    def initUI(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.title_bar = CustomTitleBar(self)
        self.main_layout.addWidget(self.title_bar)

        content_widget = QWidget()
        self.main_layout.addWidget(content_widget)
        
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(10)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter text here...")
        self.text_edit.setFont(QFont("Segoe UI", 12))
        self.text_edit.textChanged.connect(self.update_sentence_count)
        self.text_edit.setToolTip("Enter the text you want to convert to speech.")
        content_layout.addWidget(self.text_edit)

        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()

        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(QLabel("Sample Rate:"))
        self.sample_rate_input = QLineEdit()
        self.sample_rate_input.setPlaceholderText("e.g., 24000")
        self.sample_rate_input.setToolTip("Set the audio sample rate (e.g., 24000). Higher is better quality.")
        h_layout1.addWidget(self.sample_rate_input)
        h_layout1.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.voice_presets = [ f"v2/en_speaker_{i}" for i in range(10) ]
        languages = sorted(set(preset.split('/')[1][:2] for preset in self.voice_presets))
        self.language_combo.addItems(languages)
        self.language_combo.currentTextChanged.connect(self.update_voice_presets)
        self.language_combo.setToolTip("Select the language of the speaker.")
        h_layout1.addWidget(self.language_combo)
        settings_layout.addLayout(h_layout1)

        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(QLabel("Voice Preset:"))
        self.voice_preset_combo = QComboBox()
        self.voice_preset_combo.setToolTip("Select a specific voice preset for the chosen language.")
        h_layout2.addWidget(self.voice_preset_combo)
        self.sentence_count_label = QLabel("Sentences: 0")
        h_layout2.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        h_layout2.addWidget(self.sentence_count_label)
        settings_layout.addLayout(h_layout2)
        
        settings_group.setLayout(settings_layout)
        content_layout.addWidget(settings_group)

        self.waveform_widget = WaveformWidget()
        content_layout.addWidget(self.waveform_widget)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderMoved.connect(self.seek_audio)
        self.seek_slider.setToolTip("Scrub through the generated audio timeline.")
        content_layout.addWidget(self.seek_slider)

        buttons_layout = QHBoxLayout()
        self.generate_button = QPushButton("Generate Audio")
        self.generate_button.clicked.connect(self.generate_audio)
        self.generate_button.setToolTip("Generate the audio waveform from the input text.")
        self.save_button = QPushButton("Save Audio")
        self.save_button.clicked.connect(self.save_audio)
        self.save_button.setEnabled(False)
        self.save_button.setToolTip("Save the generated audio to a .wav file.")
        self.play_button = QPushButton("Play Audio")
        self.play_button.clicked.connect(self.play_audio)
        self.play_button.setEnabled(False)
        self.play_button.setToolTip("Play or stop the generated audio.")
        buttons_layout.addWidget(self.generate_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.play_button)
        content_layout.addLayout(buttons_layout)
        
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready.")
        self.loading_wheel = LoadingWheel(self)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.loading_wheel)
        content_layout.addLayout(status_layout)

        self.update_voice_presets(self.language_combo.currentText())

    def update_voice_presets(self, language):
        self.voice_preset_combo.clear()
        filtered_presets = [p for p in self.voice_presets if p.startswith(f"v2/{language}")]
        self.voice_preset_combo.addItems(filtered_presets)

    def update_sentence_count(self):
        text = self.text_edit.toPlainText()
        sentences = nltk.sent_tokenize(text) if text else []
        self.sentence_count_label.setText(f"Sentences: {len(sentences)}")
        
    def seek_audio(self, position_ms):
        if self.audio_data is None: return
        self.current_position_ms = position_ms
        self.update_waveform_signal.emit(position_ms)
        self.update_status_signal.emit(f"Seeked to {position_ms / 1000:.2f}s")

    def save_audio(self):
        if self.audio_data is None:
            self.status_label.setText("Error: No audio to save.")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Audio File", "", "WAV Files (*.wav);;All Files (*)")
            
            if file_path:
                if isinstance(self.audio_data, AudioSegment):
                    self.audio_data.export(file_path, format="wav")
                else:
                    audio_np = self.audio_data.squeeze().cpu().numpy().astype(np.float32)
                    sf.write(file_path, audio_np, self.sample_rate)
                
                self.status_label.setText(f"Audio saved to {file_path}")
        except Exception as e:
            self.handle_error(f"Error saving audio: {str(e)}\n\n{traceback.format_exc()}")

    def on_audio_generation_error(self, error_message):
        self.handle_error(f"Audio generation error: {error_message}")

    def setNightModeStyle(self):
        self.setStyleSheet("""
            TextToSpeechApp {
                background-color: #2B2B2B;
            }
            #customTitleBar {
                background-color: #1E1E1E;
            }
            #minimizeButton, #maximizeButton, #closeButton {
                background-color: transparent;
                color: #CCCCCC;
                border: none;
                font-family: "Segoe UI Symbol";
                font-size: 14px;
            }
            #minimizeButton:hover, #maximizeButton:hover { background-color: #5a5a5a; }
            #closeButton:hover { background-color: #E81123; }
            
            QWidget { 
                background-color: #2B2B2B; 
                color: #CCCCCC; 
                font-family: "Segoe UI";
            }
            QTextEdit, QLineEdit { 
                background-color: #1e1e1e; 
                color: #CCCCCC; 
                border: 1px solid #5a5a5a; 
                padding: 8px;
                border-radius: 4px;
                font-size: 11pt;
            }
            QPushButton { 
                background-color: #4A9EFF; 
                color: #FFFFFF; 
                border: none; 
                padding: 10px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #60AFFF; }
            QPushButton:pressed { background-color: #3A8EEF; }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
            QComboBox { 
                background-color: #1e1e1e; 
                border: 1px solid #5a5a5a;
                padding: 8px;
                border-radius: 4px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                 background-color: #1e1e1e;
                 selection-background-color: #4A9EFF;
                 border: 1px solid #5a5a5a;
            }
            QLabel { color: #CCCCCC; }
            QGroupBox { 
                border: 1px solid #5a5a5a; 
                margin-top: 0.5em;
                padding: 1em;
                border-radius: 4px;
            }
            QGroupBox::title { 
                color: #CCCCCC;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QGroupBox[checkable="true"] {
                padding-top: 1.5em;
            }
            QProgressBar {
                border: 1px solid #5a5a5a;
                border-radius: 4px;
                text-align: center;
                color: #CCCCCC;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk { background-color: #4a9eff; border-radius: 3px; }
            QSlider::groove:horizontal {
                border: 1px solid #5a5a5a;
                height: 4px;
                background: #1e1e1e;
                margin: 2px 0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #4a9eff;
                border: 1px solid #5a5a5a;
                width: 16px;
                height: 16px;
                margin: -7px 0;
                border-radius: 8px;
            }
            QToolTip {
                color: #FFFFFF;
                background-color: #1E1E1E;
                border: 1px solid #5a5a5a;
                padding: 5px;
                border-radius: 3px;
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextToSpeechApp()
    window.show()
    sys.exit(app.exec())
