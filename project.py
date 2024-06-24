from ultralytics import YOLOv10
import cv2
import math
import random
import time

# Function to read words from file
def read_words_from_file(file_path):
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return words

# Function to choose a new target word
def choose_new_target():
    global target_word, detected_letters
    target_word = random.choice(words)
    detected_letters = [False] * len(target_word)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load model
model = YOLOv10("model_yolo10_fix_1.pt")

# Object classes
classNames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z', 'background']

# Read target words from file
words = read_words_from_file('target_words.txt')

choose_new_target()

def detect_and_update_status(results):
    global detected_letters

    detected_classes = [int(box.cls[0]) for r in results for box in r.boxes]
    detected_labels = [classNames[cls] for cls in detected_classes]

    for i, letter in enumerate(target_word):
        if not detected_letters[i] and letter in detected_labels:
            detected_letters[i] = True

def display_message(img, message, duration=1):
    height, width, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(img, message, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
    cv2.imshow('Webcam', img)
    cv2.waitKey(int(duration * 1000))

def show_messages_sequentially(img, messages, duration=0.5):
    """Displays messages sequentially with a delay between each."""
    for message in messages:
        display_message(img.copy(), message, duration)  # Use a copy to avoid overwriting

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    detect_and_update_status(results)

    if all(detected_letters):
        # message = f"Good job! Keren!\n"
        # display_message(img, message, duration=0.5)
        choose_new_target()
        # message = f"New target:\n{target_word}"
        # display_message(img, message, duration=0.5)
        messages = [
            f"Good job! Keren!",
            f"New target: {target_word}"
        ]
        show_messages_sequentially(img, messages, duration=1)
    

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, f'{classNames[cls]} {confidence:.2f}', org, font, fontScale, color, thickness)

    display_word = ''.join([f'[{letter}]' if detected else letter for letter, detected in zip(target_word, detected_letters)])
    cv2.putText(img, f'Target Word: {display_word}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
