import cv2
import mediapipe as mp

# Initialize the MediaPipe hand model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Start the webcam feed
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        # Read a frame from the webcam feed
        ret, img = cap.read()
        
        # Convert the frame to RGB and process it with MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Check if a hand is detected
        if results.multi_hand_landmarks:
            # Get the hand landmark points for the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get the landmark point for the index finger
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_MCP = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_finger_PIP = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            index_finger_DIP = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            
            # Get the pixel coordinates of the index finger point
            height, width, _ = img.shape
            index_finger_tip_x = int(index_finger_tip.x * width)
            index_finger_tip_y = int(index_finger_tip.y * height)

            index_finger_MCP_x = int(index_finger_MCP.x * width)
            index_finger_MCP_y = int(index_finger_MCP.y * height)

            index_finger_PIP_x = int(index_finger_PIP.x * width)
            index_finger_PIP_y = int(index_finger_PIP.y * height)

            index_finger_DIP_x = int(index_finger_DIP.x * width)
            index_finger_DIP_y = int(index_finger_DIP.y * height)

            # Draw a circle around the index finger point
            cv2.circle(img, (index_finger_tip_x, index_finger_tip_y), 5, (250, 250, 0), -2)
            cv2.circle(img, (index_finger_MCP_x, index_finger_MCP_y), 5, (250, 250, 0), -2)
            cv2.circle(img, (index_finger_PIP_x, index_finger_PIP_y), 5, (250, 250, 0), -2)
            cv2.circle(img, (index_finger_DIP_x, index_finger_DIP_y), 5, (250, 250, 0), -2)
            
            cv2.putText(img, "INDEX FINGER", (index_finger_tip_x - 145 , index_finger_tip_y), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 1)
        cv2.imshow('Index Finger Detection', img)
        
        # Check if the user wants to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
