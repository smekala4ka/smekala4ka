from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5 import uic

import os
import sys
import cv2
from PIL import Image, ImageOps
import roop.globals
import roop.metadata
import numpy as np
from pydub import AudioSegment
import noisereduce
import soundfile as sf
import moviepy.editor as mp
from roop.face_analyser import get_one_face
from roop.capturer import get_video_frame, get_video_frame_total
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.predictor import predict_frame, clear_predictor
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import is_image, is_video, resolve_relative_path


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('untitled.ui', self)
        self.setWindowTitle("Jiron")
        self.radioButton.setChecked(True)

        self.face = None
        self.target = None
        self.PREVIEW_MAX_HEIGHT = 700
        self.PREVIEW_MAX_WIDTH = 1200

        self.graphicsView.setAcceptDrops(True)
        self.graphicsView.dragEnterEvent = self.dragEnterEvent
        self.graphicsView.dragMoveEvent = self.dragMoveEvent
        self.graphicsView.dropEvent = self.dropEvent

        self.graphicsView_2.setAcceptDrops(True)
        self.graphicsView_2.dragEnterEvent = self.dragEnterEvent
        self.graphicsView_2.dragMoveEvent = self.dragMoveEvent
        self.graphicsView_2.dropEvent = self.dropEvent

        self.pushButton.clicked.connect(self.getPathImg)
        self.pushButton_2.clicked.connect(self.getPathVideo)
        self.pushButton_3.clicked.connect(self.start)



        self.show()

    def getPathImg(self):
        wb_patch = QtWidgets.QFileDialog.getOpenFileName(filter = "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif)")[0]
        self.load_image_to_graphics_view(wb_patch, self.graphicsView)
        self.face = wb_patch

    def getPathVideo(self):
        wb_patch = QtWidgets.QFileDialog.getOpenFileName(filter = "Videos (*.mp4 *.avi *.mkv *.mov)")[0]
        frame_number = 0
        capture = cv2.VideoCapture(wb_patch)
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if has_frame:
            try:
                os.mkdir('./temp')
            except:
                pass
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.save('./temp/face.jpg', "JPEG")
            path = "./temp/face.jpg"
            self.load_image_to_graphics_view(path, self.graphicsView_2)
            self.target = wb_patch
            return
        capture.release()
        cv2.destroyAllWindows()
        self.load_image_to_graphics_view(wb_patch, self.graphicsView_2)
        self.target = wb_patch



    def dragEnterEvent(self, event):
        mime_data = event.mimeData()
        urls = mime_data.urls()

        if len(urls) == 1:
            file_path = urls[0].toLocalFile()

            if self.is_image(file_path) or self.is_video(file_path):
                event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        mime_data = event.mimeData()
        urls = mime_data.urls()

        if len(urls) == 1:
            file_path = urls[0].toLocalFile()

            if self.is_image(file_path):
                self.load_image_to_graphics_view(file_path, self.graphicsView)
                self.face = file_path

            elif self.is_video(file_path):
                frame_number = 0
                capture = cv2.VideoCapture(file_path)
                if frame_number:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                has_frame, frame = capture.read()
                if has_frame:
                    try:
                        os.mkdir('./temp')
                    except:
                        pass
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image.save('./temp/face.jpg', "JPEG")
                    path = "./temp/face.jpg"
                    self.load_image_to_graphics_view(path, self.graphicsView_2)
                    self.target = file_path
                    return
                capture.release()
                cv2.destroyAllWindows()
                self.load_image_to_graphics_view(file_path, self.graphicsView_2)
                self.target = file_path

    def load_image_to_graphics_view(self, file_path, graphics_view):
        scene = QGraphicsScene()
        pixmap_item = QGraphicsPixmapItem(QPixmap(file_path))
        scene.addItem(pixmap_item)
        graphics_view.setScene(scene)
        graphics_view.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def is_image(self, file_path: str) -> bool:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
        return any(file_path.lower().endswith(ext) for ext in image_extensions)

    def is_video(self, file_path: str) -> bool:
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov']
        return any(file_path.lower().endswith(ext) for ext in video_extensions)
    #
    def start(self):
        roop.globals.target_path = self.target
        roop.globals.source_path = self.face
        self.create_video()


    def create_video(self):
        video_frame_total = get_video_frame_total(roop.globals.target_path)
        i = 0
        if video_frame_total > 0:
            # Открываем видео для записи
            print(1)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            print(2)
            out = cv2.VideoWriter('../output_video.mp4', fourcc, 30, (self.PREVIEW_MAX_WIDTH, self.PREVIEW_MAX_HEIGHT))
            # Обработка каждого кадра и запись в видео
            for frame_number in range(video_frame_total):
                i += 1
                temp_frame = get_video_frame(roop.globals.target_path, frame_number)
                if predict_frame(temp_frame):
                    sys.exit()
                source_face = get_one_face(cv2.imread(roop.globals.source_path))
                if not get_face_reference():
                    reference_frame = get_video_frame(roop.globals.target_path, roop.globals.reference_frame_number)
                    reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
                    set_face_reference(reference_face)
                else:
                    reference_face = get_face_reference()
                for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
                    temp_frame = frame_processor.process_frame(source_face, reference_face, temp_frame)
                temp_frame = cv2.resize(temp_frame, (self.PREVIEW_MAX_WIDTH, self.PREVIEW_MAX_HEIGHT))
                write_text = f'Ready {i}:{video_frame_total}'
                print(write_text)
                out.write(temp_frame)

            # Закрываем видео
            

            out.release()
        temporary_audio_file = 'temporary_audio_file.mp3'
        temporary_video_file = '../output_video.mp4' # Это сгенерированное новое видео
        output_file = '../final.mp4'

        video = mp.VideoFileClip(roop.globals.target_path)
        video.audio.write_audiofile('audio_file.mp3')
        old_audio = 'audio_file.mp3'

        if self.radioButton.isChecked():
            self.change_voice(temporary_video_file, old_audio, output_file, temporary_audio_file)
        elif self.radioButton_2.isChecked():
            self.replace_audio(temporary_video_file, old_audio, output_file)
        else:
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("checkBox error")
            msg.exec_()

        msg = QMessageBox()
        msg.setWindowTitle("Success")
        msg.setText("Success")
        msg.exec_()
        print('Success')

    def replace_audio(self,video_file, modified_audio, output_file):
        # Input audio file
        audio = mp.AudioFileClip(modified_audio)
        # Input video file
        video = mp.VideoFileClip(video_file)
        # adding external audio to video
        final_video = video.set_audio(audio)
        # Extracting final output video
        final_video.write_videofile(output_file)

    def change_voice(self, temporary_video_file, old_audio, output_file, temporary_audio_file):

        # Загрузите аудиофайл
        audio = AudioSegment.from_file(old_audio)
        # Получите данные аудио в виде numpy array
        data = np.array(audio.get_array_of_samples(), dtype=np.int16)
        # Применение noisereduce для удаления фонового шума
        reduced_noise = noisereduce.reduce_noise(audio.get_array_of_samples(), audio.frame_rate)
        # Сохранение модифицированного и очищенного от фонового шума аудио в новый файл в формате MP3
        modified_audio = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=reduced_noise.dtype.itemsize,
            channels=1
        )
        modified_audio = modified_audio.speedup(playback_speed=2.0)

        modified_audio.export(temporary_audio_file, format="mp3")
        self.replace_audio(temporary_video_file, temporary_audio_file, output_file)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    sys.exit(app.exec_())
