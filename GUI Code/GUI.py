import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import calibrate
import runDetection
import firebase_admin
from firebase_admin import credentials, firestore


# Path to your Firebase service account key
cred = credentials.Certificate("..\Don't Upload\putt-master-firebase-firebase-adminsdk-fbsvc-da7f2a7f2b.json")
app = firebase_admin.initialize_app(cred)

def example_function(username):
    # Simulate work being done
    import time
    time.sleep(2)
    print(f"Function ran for user: {username}")

class FullScreenApp:
    def __init__(self, master):
        self.master = master
        self.master.title("My Fullscreen App")
        self.master.attributes("-fullscreen", True)

        # Initialize camera
        self.vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.vs.set(cv2.CAP_PROP_FPS, 60)
        self.vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Frame for spacing and organization
        self.frame = tk.Frame(master)
        self.frame.pack(expand=True)

        # Username input
        self.username_label = tk.Label(self.frame, text="Enter Username:", font=("Arial", 18))
        self.username_label.pack(pady=10)

        self.username_entry = tk.Entry(self.frame, font=("Arial", 18))
        self.username_entry.pack(pady=10)

        # Checkboxes
        self.option1_var = tk.BooleanVar()
        self.option2_var = tk.BooleanVar()
        self.option3_var = tk.BooleanVar()
        self.option4_var = tk.BooleanVar()

        self.checkbox1 = tk.Checkbutton(self.frame, text="Camera Frame", font=("Arial", 16),
                                        variable=self.option1_var)
        self.checkbox1.pack(pady=5)

        self.checkbox2 = tk.Checkbutton(self.frame, text="Corrected Frame", font=("Arial", 16),
                                        variable=self.option2_var)
        self.checkbox2.pack(pady=5)

        self.checkbox3 = tk.Checkbutton(self.frame, text="Ball Mask", font=("Arial", 16),
                                        variable=self.option3_var)
        self.checkbox3.pack(pady=5)

        self.checkbox4 = tk.Checkbutton(self.frame, text="Hole Mask", font=("Arial", 16),
                                        variable=self.option4_var)
        self.checkbox4.pack(pady=5)

        # Buttons
        self.button1 = tk.Button(self.frame, text="Run Calibration", font=("Arial", 18),
                         command=lambda: self.run_task(lambda: calibrate.main(self.vs)))
        self.button1.pack(pady=20)

        self.button2 = tk.Button(self.frame, text="Start Putt", font=("Arial", 18),
                         command=lambda: self.run_task(lambda: runDetection.cameraDetection(
                             self.vs,
                             self.option1_var.get(),
                             self.option2_var.get(),
                             self.option3_var.get(),
                             self.option4_var.get(),
                             self.username_entry.get()
                         )))
        self.button2.pack(pady=20)

        # Confirmation message
        self.confirmation_label = tk.Label(self.frame, text="", font=("Arial", 18), fg="green")
        self.confirmation_label.pack(pady=20)

        # Exit button
        self.exit_button = tk.Button(self.frame, text="Exit", font=("Arial", 18),
                                     command=self.close)
        self.exit_button.pack(pady=20)

    def run_task(self, task_func):
        username = self.username_entry.get()
        print(f"Username: {username}")
        if not username:
            messagebox.showerror("Input Error", "Please enter a username.")
            return

        # Clear previous message
        self.confirmation_label.config(text="")

        # Run the task in a separate thread so the GUI doesn't freeze
        threading.Thread(target=self._task_wrapper, args=(task_func, username)).start()

    def _task_wrapper(self, task_func, username):
        task_func()  # Run the function
        # After task completes, update the GUI
        self.confirmation_label.config(text=f"Task completed for user: {username}")

    def close(self):
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FullScreenApp(root)
    root.mainloop()
