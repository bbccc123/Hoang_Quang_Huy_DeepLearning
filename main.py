import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from ultralytics import YOLO
from gui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.Browse.clicked.connect(self.start_browse_image)
        self.thread = {}

    def start_browse_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "All Files (*);;JPEG (*.jpg;*.jpeg)", options=options)
        if file_path:
            self.thread[1] = live_stream(file_path)
            self.thread[1].signal.connect(self.show_image)
            self.thread[1].start()

    def show_image(self, cv_img):
        qt_img = convert_cv_qt(cv_img)
        self.uic.label_3.setPixmap(qt_img)


def convert_cv_qt(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(700, 500, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)

class live_stream(QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, image_path):
        self.image_path = image_path
        self.stopped = False
        super(live_stream, self).__init__()

    def stop(self):
        self.stopped = True


    def run(self):
        model = YOLO("best.pt")
        img = cv2.imread(self.image_path)
        results = model(img, show=False)
        x = 0
        y = 0
        z = 0
        for result in results:
            if self.stopped:
                break
            boxes = result.boxes.numpy()
            for box in boxes:
                if self.stopped:
                    break
                if box.cls == 0:
                    x += 1
                if box.cls == 1:
                    y += 1
                if box.cls == 2:
                    z += 1    
        cv2.putText(img, f'Head: {x}', (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (151, 157, 255), 2)
        cv2.putText(img, f'Helmet: {y}', (2, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 56, 255), 2)
        cv2.putText(img, f'Person: {y}', (2, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        self.signal.emit(img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
