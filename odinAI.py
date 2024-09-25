import sys
import json
import requests
import numpy as np
import pickle
import warnings
import torch
import markdown
import pyttsx3
from datetime import datetime
from io import BytesIO
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QTextBrowser,
    QPushButton, QSplitter, QLabel, QSizePolicy, QSlider, QComboBox, QGroupBox
)
from PyQt5.QtGui import QPixmap, QFont, QColor, QTextCharFormat, QSyntaxHighlighter, QTextCursor
from PyQt5.QtCore import Qt, QRegExp, QThread, pyqtSignal, QObject
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist, cosine
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

class VitalContextExtractor(QObject):
    finished = pyqtSignal(str, str)  # Emit both vital_context and user_input

    def __init__(self, vector_db, user_input):
        super().__init__()
        self.vector_db = vector_db
        self.user_input = user_input

    def run(self):
        input_vector = self.vector_db.encode_message(self.user_input)
        similar_messages = self.vector_db.find_similar_messages(input_vector, n=5)
        
        context_history = ""
        for msg in similar_messages:
            context_history += f"{msg['type']}: {msg['message']}\nContext: {msg['context']}\n"
            context_history += "Similar messages:\n"
            for similar_msg in msg['full_context']['similar_messages']:
                context_history += f"  {similar_msg['type']}: {similar_msg['message']}\n  Context: {similar_msg['context']}\n"
            context_history += "\n"
        
        prompt = f"""Analyze the following user input and relevant historical context. 
        Provide a concise summary that captures the essential information needed to understand and respond to the user's current query. 
        Focus on identifying recurring themes, related topics, and any information that directly pertains to the user's input.
        
        Current user input: {self.user_input}

        Relevant historical context:
        {context_history}

        Vital context summary:"""

        self.context_worker = ApiWorker("http://192.168.0.154:11434/api/chat", {
            "model": "odinai",
            "messages": [
                {"role": "system", "content": prompt}
            ],
            "stream": False,
        })
        self.context_worker.finished.connect(self.handle_context_response)
        self.context_worker.start()

    def handle_context_response(self, vital_context):
        # Update the vector database with the vital context
        for msg in self.vector_db.messages:
            if msg['message'] == self.user_input:
                msg['full_context']['vital_context'] = vital_context
                break
        
        self.finished.emit(vital_context, self.user_input)

class MessageEncoder(QObject):
    finished = pyqtSignal()

    def __init__(self, vector_db, user_input):
        super().__init__()
        self.vector_db = vector_db
        self.user_input = user_input

    def run(self):
        input_vector = self.vector_db.encode_message(self.user_input)
        self.vector_db.add_vector(input_vector, self.user_input, 'user')
        self.finished.emit()

class TTSWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, tts_engine, text):
        super().__init__()
        self.tts_engine = tts_engine
        self.text = text

    def run(self):
        self.tts_engine.say(self.text)
        self.tts_engine.runAndWait()
        self.finished.emit()

class ApiWorker(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, url, data):
        super().__init__()
        self.url = url
        self.data = data

    def preload_model(self):
        preload_url = "http://192.168.0.154:11434/api/generate"
        preload_data = {
            "model": self.data["model"],
            "keep_alive": 7200
        }
        try:
            response = requests.post(preload_url, json=preload_data)
            response.raise_for_status()
            print(f"Model {self.data['model']} preloaded successfully")
        except Exception as e:
            print(f"Error preloading model: {str(e)}")

    def run(self):
        self.preload_model()
        try:
            response = requests.post(
                self.url,
                json=self.data,
                stream=self.data.get("stream", True)
            )
            response.raise_for_status()

            if self.data.get("stream", True):
                output = ""
                for line in response.iter_lines():
                    if line:
                        body = json.loads(line)
                        if "error" in body:
                            raise Exception(body["error"])
                        if not body.get("done", False):
                            content = body.get("message", {}).get("content", "")
                            output += content
                            self.progress.emit(content)
                        else:
                            break
                self.finished.emit(output)
            else:
                content = response.json()['message']['content']
                self.finished.emit(content)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

class CodeHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        code_format = QTextCharFormat()
        code_format.setBackground(QColor("#2b2b2b"))
        code_format.setForeground(QColor("#a9b7c6"))

        code_pattern = QRegExp('```[\s\S]*?```')
        code_pattern.setMinimal(True)
        self.highlighting_rules.append((code_pattern, code_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

class SimpleVectorDatabase:
    def __init__(self, filepath=None, clustering_threshold=0.70):
        self.filepath = filepath
        self.vectors = []
        self.messages = []
        self.types = []
        self.contexts = []
        self.clustering_threshold = clustering_threshold
        self.linkage_matrix = None
        self.cluster_labels = []
        self.context_cache = {}
        self.command_session_active = False
        self.command_session_history = ""
        if self.filepath:
            self.load()

    def extract_context(self, message):
        if message in self.context_cache:
            print(f"Using cached context for message: '{message}'")
            return self.context_cache[message]
        
        try:
            # Prepare the prompt with more emphasis on preserving key information
            prompt = f"""Analyze the following message and provide a concise summary that preserves key information:

    1. Identify the main topic or intent of the message.
    2. Extract important details such as names, places, dates, and specific facts.
    3. Preserve any unique or noteworthy phrases or terminology used.
    4. Identify the overall sentiment or tone of the message.
    5. If the message is part of a conversation, note any references to previous messages or ongoing topics.

    Provide a summary that captures these elements while maintaining the essence of the original message. Aim for a balance between conciseness and information preservation.

    Message to analyze: '{message}'

    Summary:"""

            response = requests.post(
                "http://192.168.0.154:11434/api/chat",
                json={
                    "model": "odinai",
                    "messages": [{"role": "system", "content": prompt}],
                    "stream": False,
                }
            )
            response.raise_for_status()
            context = response.json()['message']['content']
            self.handle_extracted_context(message, context)
            return context
        except Exception as e:
            print(f"Error in context extraction: {e}")
            return f"Error in context extraction: {str(e)}"

    def handle_extracted_context(self, message, context):
        print(f"Extracted context for message: '{message}': {context}")
        self.context_cache[message] = context

    def encode_message(self, message, model_encoder=None, chunk_size=256):
        context = self.extract_context(message)
        full_message = f"{context} {message}"
        if model_encoder is None:
            model_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        words = full_message.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        vectors = [model_encoder.encode(chunk, show_progress_bar=False, convert_to_tensor=True) for chunk in chunks]
        vector = torch.mean(torch.stack(vectors), dim=0) if vectors else np.zeros(model_encoder.get_sentence_embedding_dimension())
        return vector.cpu().numpy()

    def add_vector(self, vector, message, msg_type, context=None):
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if message not in [msg['message'] for msg in self.messages]:
            if context is None:
                context = self.extract_context(message)
            
            # If context is still None after extraction attempt, set a default value
            if context is None:
                context = "Context extraction failed"

            print(f"\nAdding message with context: {context}")
            self.vectors.append(vector)
            self.messages.append({
                'message': message, 
                'type': msg_type, 
                'context': context, 
                'timestamp': current_timestamp,
                'full_context': {
                    'message': message,
                    'context': context,
                    'similar_messages': [],  # This will be populated in find_similar_messages
                    'vital_context': ''  # This will be populated when extracting vital context
                }
            })
            self.types.append(msg_type)
            self.contexts.append(context)
            self._update_clusters()
            self.save()
        else:
            print(f"Message '{message}' already added, skipping duplicate addition.")

    def add_interaction(self, query_vector, query_message, response_vector, response_message):
        self.add_vector(query_vector, query_message, 'query')
        self.add_vector(response_vector, response_message, 'response')
        
    def adjust_clustering_threshold(self, new_threshold):
        self.clustering_threshold = new_threshold
        self._update_clusters()

    def find_recent_similar_pairs(self, input_vector, n=5):
        if len(self.vectors) < 2:
            return []

        # Calculate cosine similarities
        similarities = [1 - cosine(input_vector, vec) for vec in self.vectors]
        
        # Create pairs of user inputs and AI responses
        pairs = []
        for i in range(0, len(self.messages) - 1, 2):
            if i + 1 < len(self.messages):
                user_msg = self.messages[i]
                ai_msg = self.messages[i + 1]
                avg_similarity = (similarities[i] + similarities[i + 1]) / 2
                pairs.append((user_msg, ai_msg, avg_similarity))

        # Sort pairs by similarity (most similar first) and select top n
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:n]        
        
    def find_similar_messages(self, vector, n=13, reduce_dims=False):
        if not self.vectors:
            return []

        vectors_for_search = self.vectors
        if reduce_dims and len(self.vectors[0]) > 50:
            pca = PCA(n_components=50)
            vectors_for_search = pca.fit_transform(self.vectors)
            vector = pca.transform([vector])[0]

        temp_vectors = np.vstack([vectors_for_search, vector])
        temp_linkage_matrix = linkage(temp_vectors, method='ward')
        temp_cluster_labels = fcluster(temp_linkage_matrix, t=self.clustering_threshold, criterion='distance')
        query_cluster = temp_cluster_labels[-1]

        cluster_indices = [i for i, label in enumerate(temp_cluster_labels[:-1]) if label == query_cluster]
        filtered_vectors = [vectors_for_search[i] for i in cluster_indices]

        if filtered_vectors:
            distances = cdist([vector], filtered_vectors, metric='cosine').flatten()
            nearest_indices = np.argsort(distances)[:n]
            similar_messages = [self.messages[cluster_indices[i]] for i in nearest_indices]

            # Update the similar_messages for each message in the result
            for msg in similar_messages:
                msg['full_context']['similar_messages'] = [
                    {
                        'message': other_msg['message'],
                        'context': other_msg['context'],
                        'type': other_msg['type'],
                        'timestamp': other_msg['timestamp']
                    }
                    for other_msg in similar_messages if other_msg != msg
                ]

            return similar_messages
        else:
            return []

    def _analyze_context_between_messages(self, messages):
        if len(messages) < 2:
            return messages

        analyzed_messages = []
        for i in range(len(messages)):
            current_message = messages[i]
            context = self._get_context_for_message(current_message, messages[:i] + messages[i+1:])
            current_message['analyzed_context'] = context
            analyzed_messages.append(current_message)

        return analyzed_messages

    def _get_context_for_message(self, message, other_messages):
        message_content = message['message']
        context_messages = " ".join([m['message'] for m in other_messages])

        prompt = f"""Analyze the context between the following message and the surrounding messages. 
        Focus on identifying relationships, common themes, and how they might be interconnected.
        Message to analyze: {message_content}
        Surrounding messages: {context_messages}
        Provide a brief summary of the contextual relationship:"""

        try:
            response = requests.post(
                "http://192.168.0.154:11434/api/chat",
                json={
                    "model": "odinai",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
            )
            response.raise_for_status()
            context_analysis = response.json()['message']['content']
        except Exception as e:
            print(f"Error in context analysis: {e}")
            context_analysis = "Error in context analysis"

        return context_analysis

    def _update_clusters(self):
        if len(self.vectors) > 1:
            self.linkage_matrix = linkage(self.vectors, method='ward')
            self.cluster_labels = fcluster(self.linkage_matrix, t=self.clustering_threshold, criterion='distance')
        else:
            self.cluster_labels = np.zeros(len(self.vectors))

    def save(self):
        with open(self.filepath, 'wb') as f:
            data = {
                'vectors': self.vectors,
                'messages': self.messages,
                'types': self.types,
                'contexts': self.contexts,
                'linkage_matrix': self.linkage_matrix,
                'cluster_labels': self.cluster_labels.tolist() if isinstance(self.cluster_labels, np.ndarray) else self.cluster_labels,
                'context_cache': self.context_cache,
                'command_session_active': self.command_session_active,
                'command_session_history': self.command_session_history
            }
            pickle.dump(data, f)

    def load(self):
        try:
            with open(self.filepath, 'rb') as f:
                data = pickle.load(f)
                self.vectors = data['vectors']
                self.messages = data['messages']
                self.types = data.get('types', [])
                self.contexts = data.get('contexts', [])
                self.linkage_matrix = data.get('linkage_matrix', None)
                self.cluster_labels = np.array(data.get('cluster_labels', []))
                self.context_cache = data.get('context_cache', {})
                self.command_session_active = data.get('command_session_active', False)
                self.command_session_history = data.get('command_session_history', "")
                if self.linkage_matrix is not None:
                    self._update_clusters()
        except (FileNotFoundError, EOFError):
            pass
        
    def plot_dendrogram(self):
        if self.linkage_matrix is not None:
            plt.figure(figsize=(10, 7), facecolor='#1e1e1e')  # Set figure background to match app theme
            ax = plt.gca()
            ax.set_facecolor('#1e1e1e')  # Set axes background to match app theme
            
            # Plot dendrogram 
            dendrogram(self.linkage_matrix, 
                    color_threshold=1, 
                    above_threshold_color='#d4d4d4',  # Light color for branches above threshold
                    leaf_font_size=8,
                    leaf_rotation=90)
            
            plt.title("Hierarchical Clustering Dendrogram", color='#d4d4d4', fontsize=16)
            plt.xlabel("Sample index", color="#d4d4d4", fontsize=12)
            plt.ylabel("Distance", color="#d4d4d4", fontsize=12)
            
            # Customize tick colors and grid
            plt.tick_params(colors='#d4d4d4', which='both')  # Change tick color
            ax.xaxis.grid(False)  # Remove x-axis grid
            ax.yaxis.grid(True, linestyle='--', color='#3e3e3e')  # Customize y-axis grid
            
            # Customize spines
            for spine in ax.spines.values():
                spine.set_color('#3e3e3e')
            
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', facecolor='#1e1e1e', edgecolor='none')
            buf.seek(0)
            plt.close()
            return buf
        else:
            return None

class ChatInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.vector_db = SimpleVectorDatabase(filepath='vector_database.pkl')
        self.global_conversation_history = ""
        self.debug_output = ""
        self.initUI()
        self.api_worker = None
        self.context_worker = None
        self.encoder_worker = None
        self.current_response = ""  

    def initUI(self):
        self.setWindowTitle('Advanced Chat Interface')
        self.setGeometry(100, 100, 1600, 900)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QTextEdit, QTextBrowser, QLineEdit, QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                padding: 5px;
            }
            QPushButton {
                background-color: #2a2a2a;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QLabel {
                color: #d4d4d4;
            }
            QSplitter::handle {
                background-color: #3e3e3e;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3e3e3e;
                height: 8px;
                background: #2d2d2d;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #d4d4d4;
                border: 1px solid #3e3e3e;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)
        
        main_layout = QHBoxLayout()
        
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (Chat)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        chat_label = QLabel('Chat Conversation')
        left_layout.addWidget(chat_label)
        
        chat_splitter = QSplitter(Qt.Vertical)
        
        self.chat_display = QTextBrowser()
        self.chat_display.setReadOnly(True)
        self.chat_display.setOpenExternalLinks(True)
        chat_splitter.addWidget(self.chat_display)
        
        # Clustering threshold control
        threshold_widget = QWidget()
        threshold_layout = QHBoxLayout(threshold_widget)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addWidget(QLabel("Clustering Sensitivity:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 200)  # Represents 0.5 to 2.0
        self.threshold_slider.setValue(70)  # Default value 1.25
        self.threshold_slider.valueChanged.connect(self.update_clustering_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        chat_splitter.addWidget(threshold_widget)
        
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)
        self.input_field = QTextEdit()
        self.send_button = QPushButton('Send')
        self.send_button.clicked.connect(self.on_send_clicked)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        chat_splitter.addWidget(input_widget)
        
        left_layout.addWidget(chat_splitter)
        
        # Middle panel (Context, Debug, and TTS)
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        
        middle_splitter = QSplitter(Qt.Vertical)
        
        self.context_display = QTextEdit()
        self.context_display.setReadOnly(True)
        context_widget = QWidget()
        context_layout = QVBoxLayout(context_widget)
        context_layout.addWidget(QLabel('Context Information'))
        context_layout.addWidget(self.context_display)
        middle_splitter.addWidget(context_widget)
        
        self.debug_display = QTextEdit()
        self.debug_display.setReadOnly(True)
        debug_widget = QWidget()
        debug_layout = QVBoxLayout(debug_widget)
        debug_layout.addWidget(QLabel('Debug and Output'))
        debug_layout.addWidget(self.debug_display)
        middle_splitter.addWidget(debug_widget)
        
        # Text-to-Speech controls
        tts_group = QGroupBox("Text-to-Speech")
        tts_layout = QVBoxLayout()
        
        self.tts_toggle = QPushButton('Enable TTS')
        self.tts_toggle.setCheckable(True)
        self.tts_toggle.toggled.connect(self.toggle_tts)
        tts_layout.addWidget(self.tts_toggle)
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.update_tts_settings)
        speed_layout.addWidget(self.speed_slider)
        tts_layout.addLayout(speed_layout)
        
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(QLabel("Pitch:"))
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setRange(0, 100)
        self.pitch_slider.setValue(50)
        self.pitch_slider.valueChanged.connect(self.update_tts_settings)
        pitch_layout.addWidget(self.pitch_slider)
        tts_layout.addLayout(pitch_layout)
        
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["Default", "Male", "Female"])
        self.voice_combo.currentTextChanged.connect(self.update_tts_settings)
        voice_layout.addWidget(self.voice_combo)
        tts_layout.addLayout(voice_layout)
        
        tts_group.setLayout(tts_layout)
        middle_splitter.addWidget(tts_group)
        
        middle_layout.addWidget(middle_splitter)
        
        # Right panel (Visualizations)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.dendrogram_label = QLabel('Dendrogram Visualization')
        self.dendrogram_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.dendrogram_label)
        
        self.graph_3d_label = QLabel('3D Memory Graph')
        self.graph_3d_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.graph_3d_label)
        
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(middle_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([500, 500, 600])
        
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)
        
        self.md = markdown.Markdown(extensions=['fenced_code', 'codehilite'])
        
        self.highlighter = CodeHighlighter(self.chat_display.document())
        
        self.chat_display.document().setDefaultStyleSheet("""
            pre {
                background-color: #2b2b2b;
                color: #a9b7c6;
                padding: 10px;
                border-radius: 5px;
                font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
                font-size: 14px;
                line-height: 1.5;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
        """)
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_enabled = False

    def toggle_tts(self, checked):
        self.tts_enabled = checked
        self.tts_toggle.setText('Disable TTS' if checked else 'Enable TTS')

    def update_tts_settings(self):
        speed = self.speed_slider.value()
        pitch = self.pitch_slider.value()
        voice = self.voice_combo.currentText()
        
        self.tts_engine.setProperty('rate', speed)
        self.tts_engine.setProperty('pitch', pitch / 50)  # Normalize pitch to 0-2 range
        
        voices = self.tts_engine.getProperty('voices')
        if voice == "Male" and len(voices) > 0:
            self.tts_engine.setProperty('voice', voices[0].id)
        elif voice == "Female" and len(voices) > 1:
            self.tts_engine.setProperty('voice', voices[1].id)
        else:
            # Use default voice
            pass

    def speak_text(self, text):
        if self.tts_enabled:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()        

    def on_send_clicked(self):
        user_input = self.input_field.toPlainText()
        if not user_input:
            return
        
        self.debug_output = ""  # Clear previous debug output
        self.add_debug_output(f"Processing user input: {user_input}")

        self.chat_display.append(f"<font color='#4CAF50'>You:</font> {user_input}")
        self.input_field.clear()

        # Create a worker thread for encoding the message
        self.encoder_worker = QThread()
        self.encoder = MessageEncoder(self.vector_db, user_input)
        self.encoder.moveToThread(self.encoder_worker)
        self.encoder_worker.started.connect(self.encoder.run)
        self.encoder.finished.connect(self.encoder_worker.quit)
        self.encoder.finished.connect(self.encoder.deleteLater)
        self.encoder_worker.finished.connect(self.encoder_worker.deleteLater)
        self.encoder.finished.connect(lambda: self.extract_vital_context(user_input))
        self.encoder_worker.start()
        
    def update_clustering_threshold(self):
        new_threshold = self.threshold_slider.value() / 100  # Convert to 0.5 - 2.0 range
        self.vector_db.adjust_clustering_threshold(new_threshold)
        self.update_visualizations()
        self.add_debug_output(f"Updated clustering threshold to {new_threshold:.2f}")

    def extract_vital_context(self, user_input):
        self.context_extractor_worker = QThread()
        self.context_extractor = VitalContextExtractor(self.vector_db, user_input)
        self.context_extractor.moveToThread(self.context_extractor_worker)
        self.context_extractor_worker.started.connect(self.context_extractor.run)
        self.context_extractor.finished.connect(self.context_extractor_worker.quit)
        self.context_extractor.finished.connect(self.context_extractor.deleteLater)
        self.context_extractor_worker.finished.connect(self.context_extractor_worker.deleteLater)
        self.context_extractor.finished.connect(self.handle_vital_context)
        self.context_extractor_worker.start()

    def handle_vital_context(self, vital_context, user_input):
        context_info = f"User Input: {user_input}\n\nVital Context:\n{vital_context}"
        self.context_display.setPlainText(context_info)
        self.add_debug_output(f"Extracted vital context for user input '{user_input}':\n{vital_context}")
        self.chat(user_input, vital_context)

    def chat(self, user_input, vital_context=""):
        self.add_debug_output(f"Chatting with user input: {user_input}")
        self.add_debug_output(f"Using vital context: {vital_context}")

        if len(self.global_conversation_history) > 5000:
            self.global_conversation_history = self.global_conversation_history[-5000:]
        self.global_conversation_history += f"\nUser: {user_input}"
        
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Retrieve similar messages from the vector database
        input_vector = self.vector_db.encode_message(user_input)
        similar_messages = self.vector_db.find_similar_messages(input_vector, n=5)
        
        # Retrieve recent similar conversation pairs
        recent_similar_pairs = self.vector_db.find_recent_similar_pairs(input_vector, n=3)
        
        # Prepare the context from similar messages
        similar_context = ""
        for msg in similar_messages:
            similar_context += f"{msg['type']}: {msg['message']}\nContext: {msg['context']}\n"
            similar_context += "Similar messages:\n"
            for similar_msg in msg['full_context']['similar_messages']:
                similar_context += f"  {similar_msg['type']}: {similar_msg['message']}\n  Context: {similar_msg['context']}\n"
            similar_context += f"Vital Context: {msg['full_context']['vital_context']}\n\n"
        
        # Prepare the context from recent similar conversation pairs, needs a tweak or two.
        recent_pairs_context = "Recent similar conversations:\n"
        for user_msg, ai_msg, similarity in recent_similar_pairs:
            recent_pairs_context += f"User: {user_msg['message']}\n"
            recent_pairs_context += f"OdinAI: {ai_msg['message']}\n"
            recent_pairs_context += f"Similarity: {similarity:.2f}\n\n"
        
        user_query = f"""Priority Instruction: Address the user's immediate question with a focused response. Incorporate all relevant contextual information provided as if it were part of your internal knowledge base. Your reply should seamlessly reflect this context as if recalling from memory, utilizing it to enhance the clarity and relevance of your answer. Do not reference the context explicitly, but apply it to inform your response effectively.

    Recent similar conversations and their contexts:
    {similar_context}

    {recent_pairs_context}

    Vital Context for current query:
    {vital_context}

    Recent Conversation History:
    {self.global_conversation_history}

    User's current query [Timestamp: {current_timestamp}]: {user_input}

    Please provide a response that addresses the user's query while incorporating relevant information from the provided context, similar conversations, and recent similar conversation pairs."""

        self.add_debug_output("\nSending the following structured message to the model for context:\n")
        self.add_debug_output(user_query)

        self.api_worker = ApiWorker("http://192.168.0.154:11434/api/chat", {
            "model": "odinai",
            "messages": [{"role": "user", "content": user_query}],
            "stream": True,
        })
        self.api_worker.progress.connect(self.update_chat_display)
        self.api_worker.finished.connect(self.handle_api_response)
        self.api_worker.start()

    def update_chat_display(self, content):
        if not self.current_response:
            self.chat_display.append(f"<font color='#2196F3'>OdinAI:</font> ")
        
        self.current_response += content
        self.chat_display.moveCursor(QTextCursor.End)
        self.chat_display.insertPlainText(content)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())
        self.add_debug_output(f"Received partial response from OdinAI: {content}")

    def handle_api_response(self, output):
        if len(self.global_conversation_history) + len(f"\nOdinAI: {self.current_response}") > 5000:
            self.global_conversation_history = self.global_conversation_history[-(5000 - len(f"\nOdinAI: {self.current_response}")):]
        self.global_conversation_history += f"\nOdinAI: {self.current_response}"

        self.add_debug_output(f"Received full response from OdinAI: {self.current_response}")
        
        response_vector = self.vector_db.encode_message(self.current_response)
        self.vector_db.add_vector(response_vector, self.current_response, 'OdinAI')
        self.add_debug_output("Added OdinAI response to vector database")

        self.update_visualizations()
        self.debug_display.setText(self.debug_output)
        
        # Speak the entire response in a separate thread
        if self.tts_enabled:
            self.tts_worker = TTSWorker(self.tts_engine, self.current_response)
            self.tts_worker.finished.connect(self.tts_worker.deleteLater)
            self.tts_worker.start()

        self.current_response = ""  # Reset the current response

    def speak_text(self, text):
        if self.tts_enabled:
            self.tts_worker = TTSWorker(self.tts_engine, text)
            self.tts_worker.finished.connect(self.tts_worker.deleteLater)
            self.tts_worker.start()

    def add_debug_output(self, message):
        self.debug_output += message + "\n"
        self.debug_display.setText(self.debug_output)
        self.debug_display.verticalScrollBar().setValue(self.debug_display.verticalScrollBar().maximum())


    def update_visualizations(self):
        self.update_dendrogram()
        self.update_3d_graph()

    def update_dendrogram(self):
        dendrogram_image = self.vector_db.plot_dendrogram()
        if dendrogram_image:
            pixmap = QPixmap()
            pixmap.loadFromData(dendrogram_image.getvalue())
            self.dendrogram_label.setPixmap(pixmap.scaled(500, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.dendrogram_label.setText("No dendrogram available")
        self.add_debug_output("Updated dendrogram visualization")

    def update_3d_graph(self):
        if len(self.vector_db.vectors) < 2:
            self.graph_3d_label.setText("Not enough data for 3D visualization")
            self.add_debug_output("Not enough data for 3D visualization")
            return

        vectors_array = np.array(self.vector_db.vectors)

        if vectors_array.shape[1] < 3:
            self.graph_3d_label.setText("Vectors don't have enough dimensions for 3D visualization")
            self.add_debug_output("Vectors don't have enough dimensions for 3D visualization")
            return

        n_samples = vectors_array.shape[0]
        perplexity = min(30, n_samples - 1)  # Adjust perplexity based on number of samples

        try:
            tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
            vectors_3d = tsne.fit_transform(vectors_array)
        except ValueError as e:
            self.graph_3d_label.setText(f"Error in t-SNE: {str(e)}")
            self.add_debug_output(f"Error in t-SNE: {str(e)}")
            return

        fig = Figure(figsize=(6, 6), facecolor='none')
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background-color:transparent;")
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('none')

        # Calculate point densities
        kde = gaussian_kde(vectors_3d.T)
        densities = kde(vectors_3d.T)

        # Normalize densities
        densities = (densities - densities.min()) / (densities.max() - densities.min())

        # Create colormap for the nebula
        colors = plt.cm.viridis(densities)
        colors[:, 3] = densities  # Set alpha based on density

        # Scatter plot with varying point sizes and colors
        scatter = ax.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2], 
                            c=colors, s=densities*100, alpha=0.6)

        # Calculate distances between all pairs of vectors
        distances = cdist(vectors_3d, vectors_3d, metric='euclidean')

        # Set a threshold for connections (you may need to adjust this)
        connection_threshold = np.percentile(distances, 5)  # Connect the closest 5% of pairs

        # Draw connections
        for i in range(len(vectors_3d)):
            for j in range(i+1, len(vectors_3d)):
                if distances[i, j] < connection_threshold:
                    color = 'gray'
                    alpha = 0.1
                    linewidth = 0.5
                    if i == len(vectors_3d) - 2 and j == len(vectors_3d) - 1:
                        # Highlight most recent connection
                        color = 'red'
                        alpha = 1.0
                        linewidth = 2
                    elif i == len(vectors_3d) - 1 or j == len(vectors_3d) - 1:
                        # Highlight connections to the most recent memory
                        color = 'yellow'
                        alpha = 0.5
                        linewidth = 1.5
                    ax.plot([vectors_3d[i, 0], vectors_3d[j, 0]],
                            [vectors_3d[i, 1], vectors_3d[j, 1]],
                            [vectors_3d[i, 2], vectors_3d[j, 2]], 
                            color=color, alpha=alpha, linewidth=linewidth)

        ax.set_title("3D Memory Connections", color='white')
        ax.set_xlabel("X", color='white')
        ax.set_ylabel("Y", color='white')
        ax.set_zlabel("Z", color='white')
        ax.tick_params(colors='white')

        ax.grid(False)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

        # Add a colorbar to show density
        cbar = fig.colorbar(scatter, ax=ax, label='Density', pad=0.1)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('Density', color='white')

        fig.tight_layout()

        if self.graph_3d_label.layout() is not None:
            while self.graph_3d_label.layout().count():
                item = self.graph_3d_label.layout().takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        if self.graph_3d_label.layout() is None:
            self.graph_3d_label.setLayout(QVBoxLayout())

        self.graph_3d_label.layout().addWidget(canvas)

        self.add_debug_output("Updated interactive 3D memory graph with nebula-like effect and connections")
        
    def add_debug_output(self, message):
        self.debug_output += message + "\n"

class MessageEncoder(QObject):
    finished = pyqtSignal()

    def __init__(self, vector_db, user_input):
        super().__init__()
        self.vector_db = vector_db
        self.user_input = user_input

    def run(self):
        input_vector = self.vector_db.encode_message(self.user_input)
        self.vector_db.add_vector(input_vector, self.user_input, 'user')
        self.finished.emit()  

def main():
    app = QApplication(sys.argv)
    ex = ChatInterface()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

