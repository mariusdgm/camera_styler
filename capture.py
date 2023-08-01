import cv2
import numpy as np
import tkinter as tk
from tkinter import colorchooser, Scale, HORIZONTAL
import dlib

class Webcam:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            print("Error opening video stream or file")
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100)
        self.kernel_size = 3
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(r'model\\shape_predictor_68_face_landmarks.dat')
    
    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def apply_background_subtraction(self, frame):
        fg_mask = self.background_subtractor.apply(frame)
        return fg_mask

    def apply_morphology(self, frame):
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        frame = cv2.erode(frame, kernel, iterations = 1)
        frame = cv2.dilate(frame, kernel, iterations = 1)
        return frame

    def apply_threshold(self, frame, threshold_value):
        _, mask = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)
        return mask

    # def apply_custom_colors(self, frame, color_black, color_white):
    #     color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    #     color_frame[frame == 0] = color_black[::-1]  # Reverse color order
    #     color_frame[frame == 255] = color_white[::-1]  # Reverse color order
    #     return color_frame

    def apply_custom_colors(self, frame, bright_areas, color_black, color_white):
        # Create a copy of the frame to avoid changing the original
        color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Color the background areas with color 1 (black in this case)
        color_frame[frame == 0] = color_black[::-1]  # Reverse color order

        # Color the non-background (foreground) areas with color 2 (white in this case)
        color_frame[frame == 255] = color_white[::-1]  # Reverse color order

        # Lastly, color the brightest areas with color 1 (black in this case)
        color_frame[bright_areas == 255] = color_black[::-1]  # Reverse color order
        
        return color_frame
    def get_face_contour(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])
            mask = np.zeros_like(frame)
            cv2.drawContours(mask, [shape], -1, (255, 255, 255), -1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            return mask
        
    def get_brightest_areas(self, frame, threshold):
        _, mask = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        return mask

    def display_frames(self, frames, window_name='Frame'):
        color_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame for frame in frames]
        all_frames = np.concatenate(color_frames, axis=1)
        cv2.imshow(window_name, all_frames)

        key = cv2.waitKey(25)
        if key == ord(' '):  # If space bar is pressed
            cv2.imwrite('captured_frame.jpg', color_frames[2])  # Save the current frame as 'captured_frame.jpg'
        if key == ord('q'):
            return False
        return True

    def release(self):
        self.cap.release()

    def close_windows(self):
        cv2.destroyAllWindows()

class App:
    def __init__(self, webcam):
        self.webcam = webcam
        self.root = tk.Tk()
        self.root.title('Settings')
        self.brightness_threshold = tk.IntVar(value=200)
        self.threshold = tk.IntVar(value=128)
        self.kernel_size = tk.IntVar(value=3)
        self.color_black = (0, 0, 0)
        self.color_white = (255, 255, 255)

        Scale(self.root, from_=0, to=255, orient=HORIZONTAL, variable=self.threshold, label='Threshold').pack()
        Scale(self.root, from_=0, to=255, orient=HORIZONTAL, variable=self.brightness_threshold, label='Brightness Threshold').pack()
        Scale(self.root, from_=1, to=21, orient=HORIZONTAL, variable=self.kernel_size, label='Kernel size', length=400).pack()
        tk.Button(self.root, text='Pick color for 0s', command=self.pick_color_black).pack()
        tk.Button(self.root, text='Pick color for 1s', command=self.pick_color_white).pack()

    def pick_color_black(self):
        color = colorchooser.askcolor()
        if color[1]:
            self.color_black = self.hex_to_rgb(color[1])

    def pick_color_white(self):
        color = colorchooser.askcolor()
        if color[1]:
            self.color_white = self.hex_to_rgb(color[1])

    def hex_to_rgb(self, hex_color):
        return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

    def main_loop(self):
        while True:
            ret, color_frame = self.webcam.get_frame()
            frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            if ret:

                # collect data from widgets
                self.webcam.kernel_size = self.kernel_size.get() // 2 * 2 + 1  # Ensure the kernel size is odd

                face_contour = self.webcam.get_face_contour(color_frame)
                if face_contour is None:  # If no face was detected, use a blank mask
                    face_contour = np.zeros_like(frame)


                bright_areas = self.webcam.get_brightest_areas(frame, self.brightness_threshold.get())

                thresholded_frame = self.webcam.apply_threshold(frame, self.threshold.get())

                background_less_frame = self.webcam.apply_background_subtraction(thresholded_frame)

                colored_frame = self.webcam.apply_custom_colors(thresholded_frame, bright_areas, self.color_black, self.color_white)
                colored_frame_with_bg = self.webcam.apply_custom_colors(background_less_frame, bright_areas, self.color_black, self.color_white)


                frames_to_display = [
                    frame,
                    thresholded_frame,
                    bright_areas,
                    colored_frame,
                    colored_frame_with_bg
                ]
                if not self.webcam.display_frames(frames_to_display):
                    break
            else:
                break
            self.root.update()
        self.webcam.release()
        self.webcam.close_windows()
        self.root.destroy()

def main():
    webcam = Webcam()
    app = App(webcam)
    app.main_loop()

if __name__ == "__main__":
    main()
