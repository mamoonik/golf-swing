import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import depthai as dai

import keyboard #load keyboard package
# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic

## Initializing mp_pose for Pose capture
#######################################################################################
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,min_tracking_confidence = 0.5 ,model_complexity=2, smooth_landmarks =True)
holistic = mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence = 0.7)

# # Setting up the Pose function.
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, model_complexity=2)
##################################################################
##################################################################
##################################################################
def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    # imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(image)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
   # Initialize a list to store the detected landmarks.
    landmarks = []
    landmarks_world = []
    # print(height, width)
    # Check if any landmarks are detected.
    if results.pose_landmarks:
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
                                  
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
            
        for landmark in results.pose_world_landmarks.landmark:
            # # Append the landmark into the list.
            landmarks_world.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
        # # Display the original input image and the resultant image.
        # plt.figure(figsize=[22,22])
        # plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        # plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return output_image, landmarks, landmarks_world

    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks, landmarks_world
def calculateAngle(landmark1, landmark2, landmark3):
    
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle
def classifyPose_Golfswing_RIGHT_SIDE_view(landmarks_0, output_image_0, landmarks_1, output_image_1, display=False):
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = '.'
    label_1 = '.'


    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles for the ROMS we are interested in.

    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks_0[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks_0[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks_0[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks_0[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks_0[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks_0[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks_0[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks_0[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks_0[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks_0[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks_0[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks_0[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks_0[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks_0[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks_0[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks_0[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks_0[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks_0[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

       ##
       ##   
       ##                        
    # Get the angle between the right hip, knee and ankle points 
    right_bending_angle = calculateAngle(landmarks_0[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks_0[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks_0[mp_pose.PoseLandmark.RIGHT_KNEE.value])
       # Get the angle between the right hip, knee and ankle points 
    left_bending_angle = calculateAngle(landmarks_0[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks_0[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks_0[mp_pose.PoseLandmark.LEFT_KNEE.value])
    


    ##
    x1,y1,z1 = landmarks_0[mp_pose.PoseLandmark.LEFT_ANKLE.value] 
    x2,y2,z2 = landmarks_0[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    mid_GROUND_x,mid_GROUND_y,mid_GROUND_z = (x1+x2)/2 , (y1+y2)/2, (z1+z2)/2

    x3,y3,z3 = landmarks_0[mp_pose.PoseLandmark.LEFT_HIP.value] 
    x4,y4,z4 = landmarks_0[mp_pose.PoseLandmark.RIGHT_HIP.value]
    mid_HIP_x,mid_HIP_y,mid_HIP_z = (x3+x4)/2 , (y3+y4)/2, (z3+z4)/2
    
    GROUND_HIP_NOSE_angle = calculateAngle((mid_GROUND_x,mid_GROUND_y,mid_GROUND_z),
                                      (mid_HIP_x,mid_HIP_y,mid_HIP_z),
                                      landmarks_0[mp_pose.PoseLandmark.NOSE.value])
    
    x5,y5,z5 = landmarks_0[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    x6,y6,z6 = landmarks_0[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_SHOULDER_x,mid_SHOULDER_y,mid_SHOULDER_z = (x5+x6)/2 , (y5+y6)/2, (z5+z6)/2

    dist_between_shoulders = round(math.sqrt((int(x5)-int(x6))**2 + (int(y5)-int(y6))**2))

    x7,y7,z7 = landmarks_0[mp_pose.PoseLandmark.NOSE.value] 
    lenght_of_body = round(math.sqrt((int(x7)-int(mid_GROUND_x))**2 + (int(y7)-int(mid_GROUND_y))**2))


    x8,y8,z8 = landmarks_0[mp_pose.PoseLandmark.LEFT_PINKY.value] 
    x9,y9,z9 = landmarks_0[mp_pose.PoseLandmark.RIGHT_PINKY.value] 

    x10,y10,z10 = landmarks_0[mp_pose.PoseLandmark.NOSE.value]

    x11,y11,z11 = landmarks_0[mp_pose.PoseLandmark.LEFT_KNEE.value]
    x12,y12,z12 = landmarks_0[mp_pose.PoseLandmark.RIGHT_KNEE.value]

    ## Drawing shoulder vertical lines
    cv2.line(output_image_0, (x1,y1), (x1,y1-300), [0,0,255], thickness = 2, lineType = cv2.LINE_8, shift = 0)
    cv2.line(output_image_0, (x2,y2), (x2,y2-300), [0,0,255], thickness = 2, lineType = cv2.LINE_8, shift = 0)

    cv2.circle(output_image_0, (round(mid_SHOULDER_x),round(mid_SHOULDER_y)), round(lenght_of_body/2), color = [128,0,0], thickness =2)

    cv2.line(output_image_0, (x3,y3), (x4,y4), [0,255,255], thickness = 4, lineType = cv2.LINE_8, shift = 0)
    cv2.line(output_image_0, (x5,y5), (x6,y6), [0,255,255], thickness = 4, lineType = cv2.LINE_8, shift = 0)

    cv2.line(output_image_0, (round(mid_SHOULDER_x),round(mid_SHOULDER_y)), (round(mid_HIP_x),round(mid_HIP_y)), [0,255,255], thickness = 4, lineType = cv2.LINE_8, shift = 0)
    cv2.line(output_image_0, (x11,y11), (x12,y12), [0,255,255], thickness = 4, lineType = cv2.LINE_8, shift = 0)

    cv2.line(output_image_0, (round(mid_HIP_x),round(mid_HIP_y)), (round(mid_HIP_x),round(mid_HIP_y-300)), [0,0,255], thickness = 2, lineType = cv2.LINE_8, shift = 0)
    
    midHip_midShoulder_angle_fromVertical = calculateAngle((mid_HIP_x,mid_HIP_y-300,mid_HIP_z),
                                      (mid_HIP_x,mid_HIP_y,mid_HIP_z),
                                      (mid_SHOULDER_x,mid_SHOULDER_y,mid_SHOULDER_z))
    
    if midHip_midShoulder_angle_fromVertical > 180:
        midHip_midShoulder_angle_fromVertical = 360 -midHip_midShoulder_angle_fromVertical
    cv2.putText(output_image_0, "Core Angle:" + str(midHip_midShoulder_angle_fromVertical), (round(mid_HIP_x)+20,round(mid_HIP_y)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,color = [0,0,255], thickness = 2)


    try:            
        # Check if person is in ADDRESS stage
        if x8/x9 >0/8  and x8/x9 < 1.2 and y8 > mid_SHOULDER_y and y9 > mid_SHOULDER_y: #Checking if both hands are on club grip
            if x8 > x2 and x8 < x1 and x9 >x2 and x9 < x1: 
                label = "ADDRESS pose established"        
                # Check if person HEAD has left the boundary of ankle vertical lines
                if x10 < x2 or x10 > x1: 
                    cv2.circle(output_image_0, (x10,y10), 20, color = [0,0,255], thickness =2)
                    # print("Keep Head posture within red line boundary")
                    cv2.putText(output_image_0, "Keep Head posture within red line boundary", (x10+20,y10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,color = [0,0,255], thickness = 2)
                    
                    ## While in ADDRESS pose, we want to check the pose correction from back view as well
                    output_image_1, label_1 = classifyPose_Golfswing_BACK_SIDE_view(landmarks_1, output_image_1, display=False)

    except:
        pass

    # Check if person is in TAKE BACK stage
    try:
        if x8/x9 >0/8  and x8/x9 < 1.2: #Checking if both hands are on club grip
            if x8 < x2 and x9 < x2: 
                label = "TAKE BACK pose in process"
                if x10 < x2 or x10 > x1: 
                    cv2.circle(output_image_0, (x10,y10), 20, color = [0,0,255], thickness =2)
                    print("Keep Head posture within red line boundary")
                    cv2.putText(output_image_0, "Keep Head posture within red line boundary", (x10+20,y10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,color = [0,0,255], thickness = 2)
    except:
        pass
    
    # Check if person has reached BACKSWING TOP
    try:
        if x8/x9 >0/8  and x8/x9 < 1.2 and y8 < mid_SHOULDER_y and y9 < mid_SHOULDER_y: #Checking if both hands are on club grip
            if x8 < x1 and x9 < x1:  #and x8 > x2 and x9 >x2 #It is not neccasary that the right and left hands be in the Red Lines defined by Left and Right ankle. It is important the Left and Right hand be left of the Left ankle and hgiher than the shoulder mid point   
                label = "BACKSWING TOP  reached. Ready to launch swing"
                if x10 < x2 or x10 > x1: 
                    cv2.circle(output_image_0, (x10,y10), 20, color = [0,0,255], thickness =2)
                    print("Keep Head posture within red line boundary")
                    cv2.putText(output_image_0, "Keep Head posture within red line boundary", (x10+20,y10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,color = [0,0,255], thickness = 2)

    except:
        pass

    # Check if person is in FOLLOW THROUGH stage
    try:
        if x8/x9 >0/8  and x8/x9 < 1.2 and y8 > mid_SHOULDER_y and y9 > mid_SHOULDER_y: #Checking if both hands are on club grip
            if x8 > x1 and x9 > x1: 
                label = "FOLLOW THROUGH stage"
    except:
        pass  

    try:
        if x8/x9 >0/8  and x8/x9 < 1.2 and y8 < mid_SHOULDER_y and y9 < mid_SHOULDER_y: #Checking if both hands are on club grip
            if x8 > x1 and x9 > x1: 
                label = "FINISH SWING"
    except:
        pass  

    # #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)   
    # Write the label on the output image. 
    cv2.putText(output_image_0, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


    ###########################################
    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image_0[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
    ######################################333
    # Return the output image and the classified label.
        return output_image_0, label, output_image_1, label_1
    
def classifyPose_Golfswing_BACK_SIDE_view(landmarks, output_image, display=False):
    '''
    This function classifies  poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = '.'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles for the ROMS we are interested in.

    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

       ##
       ##   
       ##                        
    # Get the angle between the right hip, knee and ankle points 
    right_bending_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    
       # Get the angle between the right hip, knee and ankle points 
    left_bending_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    #########################################
    ## Bringing angles to within 0 - 180 degree range

    if right_knee_angle > 180:
        right_knee_angle = 360 -right_knee_angle 

    if right_bending_angle > 180:
        right_bending_angle = 360 -right_bending_angle 
    ###############################3
    ##
    x1,y1,z1 = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] 
    x2,y2,z2 = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    mid_GROUND_x,mid_GROUND_y,mid_GROUND_z = (x1+x2)/2 , (y1+y2)/2, (z1+z2)/2

    x3,y3,z3 = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] 
    x4,y4,z4 = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    mid_HIP_x,mid_HIP_y,mid_HIP_z = (x3+x4)/2 , (y3+y4)/2, (z3+z4)/2
    
    GROUND_HIP_NOSE_angle = calculateAngle((mid_GROUND_x,mid_GROUND_y,mid_GROUND_z),
                                      (mid_HIP_x,mid_HIP_y,mid_HIP_z),
                                      landmarks[mp_pose.PoseLandmark.NOSE.value])
    
    x5,y5,z5 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] 
    x6,y6,z6 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_SHOULDER_x,mid_SHOULDER_y,mid_SHOULDER_z = (x5+x6)/2 , (y5+y6)/2, (z5+z6)/2

    dist_between_shoulders = round(math.sqrt((int(x5)-int(x6))**2 + (int(y5)-int(y6))**2))

    x7,y7,z7 = landmarks[mp_pose.PoseLandmark.NOSE.value] 
    lenght_of_body = round(math.sqrt((int(x7)-int(mid_GROUND_x))**2 + (int(y7)-int(mid_GROUND_y))**2))


    x8,y8,z8 = landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value] 
    x9,y9,z9 = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value] 

    x10,y10,z10 = landmarks[mp_pose.PoseLandmark.NOSE.value]

    x11,y11,z11 = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    x12,y12,z12 = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

 
    # # Check if leading (RIGHT) leg is straight

    #----------------------------------------------------------------------------------------------------------------
    
    cv2.line(output_image, (x12,y12), (x12,y12-300), [0,255,255], thickness = 4, lineType = cv2.LINE_8, shift = 0)

    # Check if one leg is straight
    if right_knee_angle > 165 and right_knee_angle < 179:
        # Specify the label of the pose that is tree pose.
        cv2.putText(output_image, "Leadside Knee Flexion angle" + str(180 -right_knee_angle), (x12+20,y12), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,color = [0,0,255], thickness = 2)
        label = '1. Bend your knees more'
    else:
        cv2.putText(output_image, "Leadside Knee Flexion angle" + str(180 -right_knee_angle), (x12+20,y12), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,color = [0,0,255], thickness = 2)
        label = 'Knees flexion posture CORRECT!'
    
        # Check if one leg is straight
        if right_bending_angle > 165 and right_bending_angle < 179:
            # Specify the label of the pose that is tree pose.
            cv2.putText(output_image, "Leadside Spine Flexion angle" + str(180 -right_knee_angle), (x4+20,y4), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,color = [0,0,255], thickness = 2)
            label = '2. Bend your Spine- more'
        else:
            cv2.putText(output_image, "Leadside Spine Flexion angle" + str(180 -right_knee_angle), (x4+20,y4), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,color = [0,0,255], thickness = 2)
            label = 'Spine Flexion angle posture CORRECT!'


    # #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)   
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Check if the resultant image is specified to be displayed.
    if display:
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        # Return the output image and the classified label.
        return output_image, label

#######################################################################################
#######################################################################################
# Initialize the VideoCapture object to read video recived from PT session
#############
##############################
camera_video_0 = cv2.VideoCapture(1)
camera_video_1 = cv2.VideoCapture(0)
#
##
# cap = camera_video
# camera_video_0.set(3,1280)
# camera_video_0.set(4,960)
# camera_video_1.set(3,1280)
# camera_video_1.set(4,960)

print( "Frame capture Initialized from RIGHT side and BACK side video camera")
# print("Select the camera footage you are interested in applying CV Models on: '1'for RIGHT SIDE VIEW, '2' for LEFT SIDE VIEW")            
# #listening to input
# cam_input = keyboard.read_key()
##
# cam_input = 2
##

def start_app() :
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
    