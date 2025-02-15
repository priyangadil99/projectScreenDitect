import cv2
import mediapipe as mp
import numpy as np
import random
import math

# Global variable to store the largest detected screen
largest_screen = None

# Fingertip indices in Mediapipe
INDEX_FINGER_TIP = 8

# Parameters for star animation
stars = []

def detect_hand_landmarks(frame, hands):
    """Detects hand landmarks and returns fingertip positions and their 3D coordinates."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    fingertip_positions = []
    fingertip_3d = []

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            lm = hand_landmarks.landmark[INDEX_FINGER_TIP]
            # Append 2D screen coordinates
            fingertip_positions.append((int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])))
            # Append 3D normalized coordinates (x, y, z)
            fingertip_3d.append((lm.x, lm.y, lm.z))

    return frame, fingertip_positions, fingertip_3d

def detect_screens(frame):
    """Detects and tracks the largest rectangular screen region."""
    global largest_screen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_screens = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > 3000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if 1.3 <= aspect_ratio <= 1.8:
                    detected_screens.append((x, y, w, h, area))

    if detected_screens:
        largest_screen = max(detected_screens, key=lambda screen: screen[4])

    return frame

def calculate_proximity_to_screen(screen, fingertip_3d, threshold=0.05):
    """Calculates whether the index fingertip is touching or close to the screen plane."""
    if screen is None or not fingertip_3d:
        return False

    # Screen plane definition (assuming screen is flat and in 2D for simplicity)
    screen_plane_z = 0  # Assuming the screen lies at z=0 in normalized coordinates

    x3d, y3d, z3d = fingertip_3d[0]
    distance_to_plane = abs(z3d - screen_plane_z)

    return distance_to_plane <= threshold

def render_stars(frame, center):
    """Renders very small colorful stars at the specified center."""
    global stars
    if center:
        for _ in range(5):  # Add 5 small stars on touch
            angle = random.uniform(0, 360)
            distance = random.uniform(3, 15)
            size = random.randint(2, 5)  # Smaller stars
            color = tuple(random.randint(100, 255) for _ in range(3))  # Soft pastel-like colors
            x = int(center[0] + distance * math.cos(math.radians(angle)))
            y = int(center[1] + distance * math.sin(math.radians(angle)))
            stars.append({"x": x, "y": y, "size": size, "color": color, "alpha": 255})

    # Render the stars and fade them out over time
    for star in stars:
        overlay = frame.copy()
        cv2.circle(overlay, (star["x"], star["y"]), star["size"], star["color"], -1)
        alpha = star["alpha"] / 255.0
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        star["alpha"] -= 10  # Faster fade-out

    # Remove faded-out stars
    stars = [star for star in stars if star["alpha"] > 0]

    return frame

def transform_touch_coordinates(fx, fy, screen):
    """Transforms the touch coordinates to align with the projector's display."""
    if screen is None:
        return fx, fy

    x, y, w, h, _ = screen
    # Mirror the x-coordinate relative to the screen center
    mirrored_fx = x + (w - (fx - x))

    return mirrored_fx, fy

def detect_touch_and_render_effect(frame, screen, fingertip_positions, is_touching):
    """Detects and renders stars at touch points on the screen."""
    if screen is None or not fingertip_positions:
        return frame

    fx, fy = fingertip_positions[0]
    # Transform the touch coordinates to align with the projector's display
    transformed_fx, transformed_fy = transform_touch_coordinates(fx, fy, screen)

    x, y, w, h, _ = screen
    if x <= transformed_fx <= x + w and y <= transformed_fy <= y + h and is_touching:
        frame = render_stars(frame, (transformed_fx, transformed_fy))

    return frame

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)

        # Detect hand landmarks and get fingertip positions
        frame, fingertip_positions, fingertip_3d = detect_hand_landmarks(frame, hands)

        # Detect and track the largest screen
        frame = detect_screens(frame)

        # Calculate proximity of the index fingertip to the screen
        is_touching = calculate_proximity_to_screen(largest_screen, fingertip_3d)

        # Render stars at touch points
        frame = detect_touch_and_render_effect(frame, largest_screen, fingertip_positions, is_touching)

        # Resize the frame for a larger display
        output_width = 1920  # Adjust as needed
        output_height = 1080  # Adjust as needed
        resized_frame = cv2.resize(frame, (output_width, output_height))

        cv2.imshow("Hand Touch with Tiny Stars", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()