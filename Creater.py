import cv2
import numpy as np
import os
import pyttsx3
import speech_recognition as sr
from datetime import datetime

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to get voice input with retries
def get_voice_input(prompt, retries=2):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)  # Better noise calibration
        for _ in range(retries):
            speak(prompt)
            print(f"Listening for: {prompt}...")

            try:
                audio = recognizer.listen(source, timeout=15, phrase_time_limit=8)  # Increased timeout
                response = recognizer.recognize_google(audio).strip()
                if response:
                    print(f"Recognized: {response}")  # Debugging
                    return response.lower()
            except sr.UnknownValueError:
                speak("Sorry, I didn't understand. Please try again.")
            except sr.WaitTimeoutError:
                speak("No response detected. Try again.")
            except sr.RequestError:
                speak("Speech recognition service error. Please check your internet.")

        speak("Moving forward without voice input.")
        return input(f"{prompt}: ")  # Fallback to manual input

# Get current weekday and time-based greeting
now = datetime.now()
weekday = now.strftime("%A")  
hour = now.hour
time_of_day = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"

# Welcome message
speak(f"{time_of_day}, today is {weekday}. Welcome to SVCE College.")

# Get user details
name = get_voice_input("Enter your name")
branch = get_voice_input("Enter your college branch name")

# Get ID with voice, fallback if needed
while True:
    id = get_voice_input("Enter your unique ID. Maximum four digits.")
    if id.isdigit() and len(id) <= 4:
        break
    speak("Invalid ID. Please enter a numeric ID with up to four digits.")
    id = input("Enter Unique ID (max 4 digits): ")  # Fallback to manual input

speak(f"Hello {name} from {branch}. I will now collect your face images.")

# Load Haar cascade for face detection
cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    speak("Error! Haar cascade file not found.")
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    speak("Error! Could not access the webcam.")
    exit()

# Create Data directory if not exists
os.makedirs("Data", exist_ok=True)

# Save user details
with open("datatext.txt", "a") as f:
    f.write(f"{id} {name} {branch}\n")

val = 0  # Image count

while val < 50:
    status, img = cap.read()
    if not status:
        speak("Error accessing the webcam.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        speak("No face detected. Please adjust your position.")

    for (x, y, w, h) in faces:
        val += 1
        face_img = gray[y:y+h, x:x+w]  # Extract face region
        
        # Save with error handling
        img_filename = f"Data/{id}_{val}.jpg"
        try:
            cv2.imwrite(img_filename, face_img)
        except Exception as e:
            print(f"Error saving image: {e}")

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if val % 10 == 0:  # Announce progress every 10 images
            speak(f"{val} images captured.")

    cv2.imshow('FaceDetect', img)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        speak("Exiting as per user request.")
        break

speak("Face data collection complete. Exiting now.")

cap.release()
cv2.destroyAllWindows()
