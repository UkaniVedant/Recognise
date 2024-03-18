import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
import face_recognition
import cv2
import numpy as np

kivy.require('2.0.0')


class FaceDetectionApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # OpenCV video capture
        self.video_capture = cv2.VideoCapture('http://192.168.1.100:8080/video')
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Set width to 640 pixels
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480 pixels
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)

        # Load known face encodings and names
        self.load_known_faces()

        # Clock scheduling to call update method periodically
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return self.layout

    def load_known_faces(self):
        # Load known face encodings and names
        self.known_face_encodings = []
        self.known_face_names = []

        # Load images and encodings
        known_faces_data = [
            {"name": "Rutvik", "image_path": "images/Rutvik.jpg"},
            {"name": "Vedant", "image_path": "images/vedant.jpg"},
            {"name": "Renish", "image_path": "images/renish.jpeg"},
            {"name": "Vrushti", "image_path": "images/Vrushti.jpeg"},
            {"name": "Kaka", "image_path": "images/Kaka.jpeg"},
            {"name": "Aunty", "image_path": "images/Aunty.jpeg"},
            {"name": "Papa", "image_path": "images/Papa.jpeg"},
        ]

        for data in known_faces_data:
            # Load image
            image = face_recognition.load_image_file(data["image_path"])
            
            # Compute face encoding
            face_encoding = face_recognition.face_encodings(image)
            
            # If there is no face detected or multiple faces detected, handle accordingly
            if len(face_encoding) != 1:
                print(f"Error processing {data['name']}'s image")
                continue
            
            # Append the face encoding and name to the respective lists
            self.known_face_encodings.append(face_encoding[0])
            self.known_face_names.append(data["name"])

    def update(self, dt):
        ret, frame = self.video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare face encodings with known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            face_names.append(name)

        # Draw rectangles and labels on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Convert frame to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        # Clear the canvas and add a Rectangle instruction with the texture
        self.layout.canvas.clear()
        self.layout.canvas.add(Rectangle(texture=texture, size=(frame.shape[1], frame.shape[0]), pos=self.layout.pos))

    def on_stop(self):
        # Release video capture when the app is stopped
        self.video_capture.release()


if __name__ == '__main__':
    FaceDetectionApp().run()
