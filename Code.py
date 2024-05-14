import face_recognition
import cv2
import numpy as np
import os
import time
import math
import logging
import datetime

# Function to recognize and log recognized names
def recognize_and_log(name):
    """Checks if a name matches names from the 'Naamlijst.txt' file and logs it accordingly."""
    with open("Gezichten/Naamlijst.txt", 'r') as file: 
        names_to_check = file.read().splitlines()

    if name in names_to_check:
        with open("log.txt", "a") as f:
            f.write(f"{name} recognized at {datetime.datetime.now()}\n")
    else:
        with open("log.txt", "a") as f:
            f.write(f"Unknown recognized at {datetime.datetime.now()}\n")

# Function to select camera
def select_camera():
    print("Select camera:")
    print("1. Default (0)")
    print("2. Other")
    print("Waiting for response...")
    start_time = time.time()
    while time.time() - start_time < 5:  # Wait for 5 seconds for user input
        if cv2.waitKey(1) & 0xFF == ord('1'):  # If user selects default (1)
            print("Default camera selected.")
            return 0
        elif cv2.waitKey(1) & 0xFF == ord('2'):  # If user selects other (2)
            camera_index = input("Enter camera index: ")
            print(f"Selected camera index: {camera_index}")
            return int(camera_index)
    print("No response in 5 seconds. Defaulting to option 1...")
    return 0

# Get the webcam
camera_index = select_camera()
video_capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if not(video_capture is None or not video_capture.isOpened()):
    print("\nWebcam successfully found!")
elif video_capture is None or not video_capture.isOpened():
    print("\nWebcam not found! Please check your camera connection.")
    exit()

# FPS
camera_fps_start_time = time.time()
camera_fps = 0 
camera_fps_display_time = 0
frame_count = 30
frame_time = 0.1

# Naam- en gezichtslijst
known_face_encodings = []

# Ga door alle gezichtafbeeldingen in de folder 'Gezichten' en analyseer ze
face_analysis_number = 1
while True:
    if not(os.path.exists("Gezichten/"+str(face_analysis_number)+".png")):
        break
    face_image = face_recognition.load_image_file("Gezichten/"+str(face_analysis_number)+".png")
    face_encoding = face_recognition.face_encodings(face_image)[0]
    if len(face_encoding) == 0:
        print(f"No face encoding found for image {face_analysis_number}.png")
    else:
        known_face_encodings.append(face_encoding)
        print(f"Face encoding loaded successfully for image {face_analysis_number}.png")
    face_analysis_number = face_analysis_number + 1

# Naamlijst lezen en in array met namen plaatsen
with open("Gezichten/Naamlijst.txt", 'r') as file: 
    known_face_names = file.read().splitlines()

# Waarden initializeren
face_locations = []
face_encodings = []
face_names = []
camera_size = 1
font = cv2.FONT_HERSHEY_DUPLEX
face_match_threshold = 0.8

# Programmaloop
while True:
    # Voor FPS
    frame_count = frame_count + 15
    frame_time = time.time() - camera_fps_start_time
    if frame_time >= camera_fps_display_time and frame_time != 0:
        camera_fps = int(frame_count / frame_time)
        frame_count = 0
        camera_fps_start_time = time.time()

    # Lees videoframe
    ret, frame = video_capture.read()
    print("Frame read status:", ret)  # Check the status of reading the frame

    # Als het frame niet correct is gelezen, sla de rest van de loop over
    if not ret:
        continue

    # Frameresolutie
    small_frame = cv2.resize(frame, (0, 0), fx=camera_size, fy=camera_size)

    # BGR naar RGB kleur
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    # Alle gezichten in het huidige frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Check of gezicht bestaat in de files
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Onbekend"

        # # Als er een match is, gebruik het eerste gezicht in de files
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Of gebruik het gezicht dat er het meest op lijkt
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
            # Calculate confidence percentage
            confidence_percentage = int((1 - face_distances[best_match_index]) * 100)

        face_names.append(name)

    # Frames per seconde
    cv2.putText(frame, str(camera_fps), (16, 32), font, 1.0, (0, 255, 0), 1)

    # Laat resultaten zien
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Streamresolutie
        top *= int(1 / camera_size)
        right *= int(1 / camera_size)
        bottom *= int(1 / camera_size)
        left *= int(1 / camera_size)

        # Teken rechthoek om gezicht
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Naamlabel bij de rechthoek met zekerheidswaarde erbij
        cv2.putText(frame, name, (left, bottom + 48), font, 1.0, (0, 255, 0), 1)
        if name != "Onbekend":
            cv2.putText(frame, str(confidence_percentage) + " % zekerheid", (left, bottom + 24), font, 1.0, (0, 255, 0), 1)

    # Laat videostream zien
    cv2.imshow('Video', frame)

    # Druk op toets om af te sluiten
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Afsluiten
video_capture.release()
cv2.destroyAllWindows()
