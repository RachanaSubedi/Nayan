from hashlib import new
from kivymd.app import MDApp
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.button import Button, ButtonBehavior
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
from aztec import get_images
from aztec import get_disease
import numpy as np
import sys
import main
Window.size = (1000, 600)

disease_name = ""


class ImageButton(ButtonBehavior, Image):
    pass


class DemoScreen(Screen):
    pass


class HomeScreen(Screen):
    pass


class MonitorScreen(Screen):

    def on_enter(self, *args):
        self.cap = cv2.VideoCapture(0)
        self.schedule = Clock.schedule_interval(self.update_frames, 1.0 / 30.0)

    def go_release(self):
        self.cap.release()

    def update_frames(self, event):
        ret, frame = self.cap.read()
        if(ret):
            temp = get_images(frame)
            temp_list = []
            for x in temp:
                buff = cv2.flip(x, 0).tostring()
                img_texture = Texture.create(
                    size=(x.shape[1], x.shape[0]), colorfmt='bgr')
                img_texture.blit_buffer(
                    buff, colorfmt='bgr', bufferfmt='ubyte')
                temp_list.append(img_texture)
            self.ids.img_camera.texture = temp_list[0]
            self.ids.img_face.texture = temp_list[2]
            self.ids.img_left.texture = temp_list[4]
            self.ids.img_right.texture = temp_list[3]
            self.ids.img_final.texture = temp_list[1]
            self.ids.img_blink.texture = temp_list[5]
            self.ids.img_depth.texture = temp_list[6]


class CheckScreen(Screen):

    def on_enter(self, *args):
        self.cap = cv2.VideoCapture(0)
        self.schedule = Clock.schedule_interval(self.update_frames, 1.0 / 30.0)

    def go_release(self):
        self.cap.release()

    def update_frames(self, event):
        ret, frame = self.cap.read()
        if(ret):
            temp = get_images(frame)
            global temp_list
            temp_list = []
            for x in temp:
                buff = cv2.flip(x, 0).tostring()
                img_texture = Texture.create(
                    size=(x.shape[1], x.shape[0]), colorfmt='bgr')
                img_texture.blit_buffer(
                    buff, colorfmt='bgr', bufferfmt='ubyte')
                temp_list.append(img_texture)
            self.ids.img_camera.texture = temp_list[0]
            self.ids.img_face.texture = temp_list[2]
            self.ids.img_left.texture = temp_list[4]
            self.ids.img_right.texture = temp_list[3]

    def check_disease(self):
        ret, frame = self.cap.read()
        temp = get_images(frame)

        original_image, final_image, face_crop, eye_left, eye_right, blink_graph, _ = get_images(
            frame)
        global disease_name
        disease_name = get_disease(eye_right)
        self.ids.disease_text.text = disease_name

    def check_disease_refresh(self):
        global disease_name
        disease_name = ""
        self.ids.disease_text.text = disease_name


class AboutScreen(Screen):
    pass


class MainApp(MDApp):

    def go_forward(self):
        sm = self.root.ids['screenmanager']
        sm.transition = SlideTransition(direction="left")
        sm.current = 'demoscreen'

    def go_backward(self):
        sm_back = self.root.ids['screenmanager']
        sm_back.transition = SlideTransition(direction="right")
        sm_back.current = 'homescreen'

    def go_demo(self):
        sm = self.root.ids['screenmanager']
        sm.transition = SlideTransition(direction="right")
        sm.current = 'demoscreen'

    def go_about(self):
        sm_about = self.root.ids['screenmanager']
        sm_about.transition = SlideTransition(direction="left")
        sm_about.current = 'aboutscreen'

    def go_monitor(self):
        sm_check = self.root.ids['screenmanager']
        sm_check.transition = SlideTransition(direction="left")
        sm_check.current = 'monitorscreen'

    def go_check(self):
        sm_check = self.root.ids['screenmanager']
        sm_check.transition = SlideTransition(direction="left")
        sm_check.current = 'checkscreen'


MainApp().run()
