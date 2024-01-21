from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import math
import sys
import os
import cv2
from cvzone.PoseModule import PoseDetector
from im import Login
from cvzone.HandTrackingModule import HandDetector
from SQLDB import SQLDB
from SQLDB.modules import UserTable
try:
    from cv2.version import ci_build, headless
    ci_and_not_headless = ci_build and not headless
except:
    pass
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_FONTDIR")
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

MODE = "start"
def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis
class VideoThread1(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detector = PoseDetector()

    def __init__(self):
        super().__init__()
        self.result = {}
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture("bok.mp4")
        while self._run_flag:
            ret, cv_img = cap.read()
            if not ret:
                cap = cv2.VideoCapture("bok.mp4")
                ret, cv_img = cap.read()
            cv_img = self.detector.findPose(cv_img, draw=False)
            lm_list3, box_info = self.detector.findPosition(cv_img, bboxWithHands=True, draw=False)
            try:
                sa = distanceCalculate(lm_list3[25][0:2], lm_list3[27][0:2])

                sb = distanceCalculate(lm_list3[23][0:2], lm_list3[25][0:2])
                sc = distanceCalculate(lm_list3[23][0:2], lm_list3[27][0:2])
                ugl = (sa**2 + sb**2 - sc**2) / (2 * sa * sb)
                ugl = math.acos(ugl) * (180 / math.pi)
                if ugl >= 120:
                    cv_img = cv2.line(cv_img, lm_list3[25][0:2], lm_list3[27][0:2], (0, 255, 0), 2)
                    cv_img = cv2.line(cv_img, lm_list3[23][0:2], lm_list3[25][0:2], (0, 255, 0), 2)
                    cv_img = cv2.line(cv_img, lm_list3[23][0:2], lm_list3[27][0:2], (0, 255, 0), 2)
                else:
                    cv_img = cv2.line(cv_img, lm_list3[25][0:2], lm_list3[27][0:2], (0, 0, 255), 2)
                    cv_img = cv2.line(cv_img, lm_list3[23][0:2], lm_list3[25][0:2], (0, 0, 255), 2)
                    cv_img = cv2.line(cv_img, lm_list3[23][0:2], lm_list3[27][0:2], (0, 0, 255), 2)
                self.result["ugl"] = ugl
            except IndexError as err:
                self.result["ugl"] = None
            self.change_pixmap_signal.emit(cv_img)

        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class VideoThread2(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detector = PoseDetector()
    detector_hand = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    def __init__(self, otn):
        super().__init__()
        self.mode = "start"
        self.otn = otn
        self._run_flag = True
        self.result = {}

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            mode = self.mode
            ret, cv_img = cap.read()
            if not ret:
                cap = cv2.VideoCapture("nak.mp4")
                ret, cv_img = cap.read()

            lower_range = np.array((11, 18, 124))
            upper_range = np.array((29, 46, 156))
            hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_range, upper_range)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                for contour in contours:
                    # peri = cv2.arcLength(contour, True)
                    # approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    area = int(rect[1][0] * rect[1][1])
                    if area > 30000:
                        x, y, w, h = cv2.boundingRect(contour)
                        self.result["box_height"] = h
                        self.result["box_y"] = y

                        cv2.drawContours(cv_img,[box],0,(255,0,255),2)
                        break
                else:
                    self.result["box_height"] = None
                    self.result["box_y"] = None
            if mode == "fix":
                hands, cv_img = self.detector_hand.findHands(cv_img, draw=True, flipType=True)
                if hands:
                    hand1 = hands[0]  # Get the first hand detected
                    lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
                    # bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
                    # center1 = hand1['center']  # Center coordinates of the first hand
                    # handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")
                    mx_size = max(lmList1, key=lambda cor: cor[0:2][1])[1]

                    fingers1 = self.detector_hand.fingersUp(hand1)
                    if len(hands) == 2:
                        hand2 = hands[1]
                        lmList2 = hand2["lmList"]
                        # bbox2 = hand2["bbox"]
                        # center2 = hand2['center']
                        # handType2 = hand2["type"]
                        mx_size = max(mx_size, max(lmList2, key=lambda cor: cor[0:2][1])[1])
                        # fingers2 = self.detector_hand.fingersUp(hand2)
                    self.result["mx_y"] = mx_size
                else:
                    self.result["mx_size"] = None
                    self.change_pixmap_signal.emit(cv_img)
            else:

                cv_img = self.detector.findPose(cv_img, draw=False)
                lm_list3, box_info = self.detector.findPosition(cv_img, bboxWithHands=True, draw=False)
                try:
                    length = distanceCalculate(lm_list3[28][0:2], lm_list3[27][0:2])
                    h = self.result.get("box_height")
                    if h is not None:
                        if 10 <= (self.otn / h) * length <= 15:
                            cv_img = cv2.line(cv_img, lm_list3[28][0:2], lm_list3[27][0:2], (0, 255, 0), 2)
                        else:
                            cv_img = cv2.line(cv_img, lm_list3[28][0:2], lm_list3[27][0:2], (0, 0, 255), 2)
                    else:
                        cv_img = cv2.line(cv_img, lm_list3[28][0:2], lm_list3[27][0:2], (0, 255, 0), 2)
                    self.result["legs"] = length
                except IndexError as err:
                    self.result["legs"] = None

            self.result["mode"] = mode


            self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self, data=None):
        super().__init__()
        self.data = data
        self.otn = None
        self.res = None
        self.mode = "start"
        self.timer_i = QTimer(self)
        self.setWindowTitle("Qt live label demo")
        self.disply_width = 400
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.start = QPushButton(self)
        self.login = Login(self)
        self.start.setText("старт")
        self.start.clicked.connect(self.start_event)
        self.fix = QPushButton(self)
        self.fix.clicked.connect(self.fix_event)
        self.fix.setText("зафиксировать")
        self.fix.hide()
        self.start.setDisabled(True)
        self.text_label = QLabel(self)
        self.text_label.setText("OK")
        self.text_label.setStyleSheet("color: green;")
        self.text_label.err = set()
        self.image_label2 = QLabel(self)
        self.image_label2.resize(self.disply_width, self.display_height)

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout(self)
        vbox_image = QHBoxLayout()
        vbox.addLayout(vbox_image)
        vbox_image.addWidget(self.image_label)
        vbox_image.addWidget(self.image_label2)
        vbox.addWidget(self.text_label)
        vbox.addWidget(self.start)
        vbox.addWidget(self.fix)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)
        self.move(self.rect().center())
        self.setup()
        self.moveEvent = self.move_event
        self.login.push_button.clicked.connect(self.user_data)


    def user_data(self):
        name = self.login.name.text()
        passport = self.login.passport.text()
        self.otn = int(self.login.size_box.text())
        self.data = {"name": name, "passport": passport}
        self.setup()
    def move_event(self, eve):
        if self.login.isVisible():
            self.login.move(self.x() + self.width() // 2 - self.login.width() // 2, self.y() + self.height() // 2 - self.login.height() // 2)

    def setup(self):
        if self.data is not None:
            if self.login.isVisible():
                self.login.close()
            # create the video capture thread
            self.thread1 = VideoThread1()
            # connect its signal to the update_image slot
            self.thread1.change_pixmap_signal.connect(self.update_image1)
            # start the thread
            self.thread1.start()
            self.thread2 = VideoThread2(self.otn)
            # connect its signal to the update_image slot
            self.thread2.change_pixmap_signal.connect(self.update_image2)
            self.timer_i.timeout.connect(self.pr_result)
            # start the thread
            self.thread2.start()
            self.setDisabled(False)
        else:
            self.setDisabled(True)
            img = QImage("camera.png")
            pix = QPixmap.fromImage(QImage("camera.png")).scaled(self.disply_width, self.display_height)
            self.image_label.setPixmap(pix)
            self.image_label2.setPixmap(pix)
            self.login.show()



    def closeEvent(self, event):
        self.thread1.stop()
        event.accept()

    def start_event(self):
        self.start.hide()
        self.mode = "fix"
        self.thread2.mode = "fix"
        self.fix.show()

    def fix_event(self):
        self.timer_i.start(2 * 1000)

    def pr_result(self):
        user = db.session.query(UserTable).filter(UserTable.name == self.data["name"]).first()
        if user is None:
            new_user = UserTable(name=self.data["name"], passport=self.data["passport"], result=self.res)
            db.session.add(new_user)
        else:
            user.result = self.res
        db.session.commit()
        self.timer_i.stop()
    def error_event(self):

        if self.text_label.err:
            self.res = None
            self.timer_i.stop()
            if self.mode == "fix":
                self.fix.hide()
                self.start.show()
                self.mode = "start"
                self.thread2.mode = "start"
                err = ", ".join(sorted(self.text_label.err))
                self.text_label.err.clear()
                QMessageBox.about(self, "ERROR", err)
            self.start.setDisabled(True)
            self.text_label.setText("Ошибки: " + ", ".join(sorted(self.text_label.err)))
            self.text_label.setStyleSheet("color: red;")
        else:
            self.start.setDisabled(False)
            self.text_label.setText("OK")
            self.text_label.setStyleSheet("color: green;")

    @pyqtSlot(np.ndarray)
    def update_image1(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img).scaled(self.disply_width, self.display_height)
        self.image_label.setPixmap(qt_img)
        if self.thread1.result["ugl"] is None:
            self.text_label.err.add("нет ног")
            self.error_event()
        else:
            self.text_label.err.discard("нет ног")
            self.error_event()
            if self.thread1.result["ugl"] >= 120:
                self.text_label.err.discard("ноги не ровные")
                self.error_event()
            else:
                self.text_label.err.add("ноги не ровные")
                self.error_event()



    @pyqtSlot(np.ndarray)
    def update_image2(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img).scaled(self.disply_width, self.display_height)
        self.image_label2.setPixmap(qt_img)
        if self.thread2.result.get("mode") == "fix":
            box_height = self.thread2.result.get("box_height", None)
            mx_y = self.thread2.result.get("mx_y", None)
            box_y = self.thread2.result.get("box_y", None)
            if box_height is None:
                self.text_label.err.add("нет ориентира")
                self.error_event()
                return
            else:
                self.text_label.err.discard("нет ориентира")
                self.error_event()
            if mx_y is None:
                self.text_label.err.add("нет рук")
                self.error_event()
                return
            else:
                self.text_label.err.discard("нет рук")
                self.error_event()
            pix_sm = self.otn / box_height
            self.res = (mx_y - box_y) * pix_sm
        else:
            box_height = self.thread2.result.get("box_height", None)
            length = self.thread2.result.get("legs", None)
            if box_height is None:
                self.text_label.err.add("нет ориентира")
                self.error_event()
                return
            else:
                self.text_label.err.discard("нет ориентира")
                self.error_event()
            if length is None:
                self.text_label.err.add("нет ног")
                self.error_event()
                return
            else:
                self.text_label.err.discard("нет ног")
                self.error_event()
            pix_sm = self.otn / box_height
            res = length * pix_sm
            if res > 15 or res < 10:
                self.text_label.err.add("ноги расположены неправильно")
                self.error_event()
            else:
                self.text_label.err.discard("ноги расположены неправильно")
                self.error_event()


    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    db = SQLDB("result.db")
    db.session.create_all()
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())