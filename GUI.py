import tkinter as tk
from tkinter import messagebox
import threading

# Example external function to run
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

        # Frame for spacing and organization
        self.frame = tk.Frame(master)
        self.frame.pack(expand=True)

        # Username input
        self.username_label = tk.Label(self.frame, text="Enter Username:", font=("Arial", 18))
        self.username_label.pack(pady=10)

        self.username_entry = tk.Entry(self.frame, font=("Arial", 18))
        self.username_entry.pack(pady=10)

        # Buttons
        self.button1 = tk.Button(self.frame, text="Run Task 1", font=("Arial", 18),
                                 command=lambda: self.run_task(example_function))
        self.button1.pack(pady=20)

        self.button2 = tk.Button(self.frame, text="Run Task 2", font=("Arial", 18),
                                 command=lambda: self.run_task(example_function))
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
        if not username:
            messagebox.showerror("Input Error", "Please enter a username.")
            return

        # Clear previous message
        self.confirmation_label.config(text="")

        # Run the task in a separate thread so the GUI doesn't freeze
        threading.Thread(target=self._task_wrapper, args=(task_func, username)).start()

    def _task_wrapper(self, task_func, username):
        task_func(username)  # Run the function
        # After task completes, update the GUI
        self.confirmation_label.config(text=f"Task completed for user: {username}")

    def close(self):
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FullScreenApp(root)
    root.mainloop()
