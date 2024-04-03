import tkinter as tk
import pygsheets
from threading import Thread


from GolfSwingAnalysis_v2  import *

def save_data_to_google_sheets(data):
    gc = pygsheets.authorize(service_file='credentials.json')
    sheet = gc.open('Golf Swing Analysis_New York_location1')
    worksheet = sheet.sheet1
    worksheet.append_table(values=[list(data.values())], end='A1', dimension='ROWS', overwrite=False)



class App:
    def __init__(self, master):
        self.master = master
        master.title("PythonGuides")
        master.geometry("500x300")

        self.first_name = tk.Entry(master)
        self.last_name = tk.Entry(master)
        self.email = tk.Entry(master)
        self.phone = tk.Entry(master)
        self.city = tk.Entry(master)
        self.state = tk.Entry(master)

        tk.Label(master, text="First Name").pack()
        self.first_name.pack()
        tk.Label(master, text="Last Name").pack()
        self.last_name.pack()
        tk.Label(master, text="Email").pack()
        self.email.pack()
        tk.Label(master, text="Phone").pack()
        self.phone.pack()
        tk.Label(master, text="City").pack()
        self.city.pack()
        tk.Label(master, text="State").pack()
        self.state.pack()

        self.start_button = tk.Button(master, text="Start Session", command=self.start_session, state=tk.DISABLED)
        self.start_button.pack()

        self.end_button = tk.Button(master, text="End Session", command=self.end_session, state=tk.DISABLED)
        self.end_button.pack()

        self.update_start_button_state()

        self.first_name.bind("<KeyRelease>", self.update_start_button_state)
        self.last_name.bind("<KeyRelease>", self.update_start_button_state)
        self.email.bind("<KeyRelease>", self.update_start_button_state)
        self.phone.bind("<KeyRelease>", self.update_start_button_state)
        self.city.bind("<KeyRelease>", self.update_start_button_state)
        self.state.bind("<KeyRelease>", self.update_start_button_state)

        self.start_cv_app_thread = Thread(target=self.run_cv_application)
        self.start_cv_app_thread.start()

    
    
    def update_start_button_state(self, event=None):
        user_data_values = list(self.get_user_data().values())
        # print(f"User data values: {user_data_values}")
        if all(user_data_values):
            # print("Enabling Start Session button")
            self.start_button["state"] = tk.NORMAL
        else:
            # print("Disabling Start Session button")
            self.start_button["state"] = tk.DISABLED

    def get_user_data(self):
        return {
            "first_name": self.first_name.get(),
            "last_name": self.last_name.get(),
            "email": self.email.get(),
            "phone": self.phone.get(),
            "city": self.city.get(),
            "state": self.state.get()
        }

    def start_session(self):
        self.start_button["state"] = tk.DISABLED

        user_data = self.get_user_data()
        # Save user_data to Google Sheets
        try:
            save_data_to_google_sheets(user_data)
            print("Session ended. User data saved to Google Sheets:", user_data)
        except Exception as e:
            print(f"Error saving data to Google Sheets: {e}")


    

    def run_cv_application(self):
        self.end_button["state"] = tk.NORMAL

        camera_video_0 = cv2.VideoCapture(1)
        camera_video_1 = cv2.VideoCapture(0)

        with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
            # Iterate until the webcam is accessed successfrully.
            while camera_video_0.isOpened() and camera_video_1.isOpened():
                # Read a frame.
                ok, frame_0 = camera_video_0.read()
                ok, frame_1 = camera_video_1.read()
                # Check if frame is not read properly.
                if not ok:
                    continue
                frame_height, frame_width, _ =  frame_0.shape
                # Resize the frame while keeping the aspect ratio.
                frame_0 = cv2.resize(frame_0, (int(frame_width * (640 / frame_height)), 640))
                frame_1 = cv2.resize(frame_1, (int(frame_width * (640 / frame_height)), 640))
                frame_final_0 = frame_0
                frame_final_1 = frame_1

                # # Perform Pose landmark detection.
                # Check if frame is not read properly.
                if not ok:
                    # Continue to the next iteration to read the next frame and ignore the empty camera frame.
                    continue
                #################################################
                #################################################
                
                # if cam_input=='1':        
                frame_0, landmarks_0, landmarks_world = detectPose(frame_0, pose_video, display=False)
                frame_1, landmarks_1, landmarks_world = detectPose(frame_1, pose_video, display=False)

                if landmarks_0 and landmarks_1:
                    frame_final_0, label_0, frame_final_1, label_1 = classifyPose_Golfswing_RIGHT_SIDE_view(landmarks_0, frame_0,landmarks_1, frame_1, display=False)
                else:
                    continue
            
                    

                if cv2.waitKey(1) & 0xFF==ord('q'): ## EXTRACT THE LABEL OF THE ANGLE MEASUREMENT AT A PARTICULAR FRAME
                        # breakw
                    print(label_0)
                    print(label_1)            
                    #returns the value of the LABEL when q is pressed
        #########################################################################################################


                stream_final_img = cv2.hconcat([frame_final_0, frame_final_1])
                cv2.imshow('Combined Video', stream_final_img)

                k = cv2.waitKey(1) & 0xFF  
                    # Check if 'ESC' is pressed.
                if(k == 27):    
                    # Break the loop.
                    break
        camera_video_0.release()
        camera_video_1.release()


    def end_session(self):
        
        """Clear all the input fields."""
        self.first_name.delete(0, 'end')
        self.last_name.delete(0, 'end')
        self.email.delete(0, 'end')
        self.phone.delete(0, 'end')
        self.city.delete(0, 'end')
        self.state.delete(0, 'end')

root = tk.Tk()
app = App(root)
root.mainloop()
