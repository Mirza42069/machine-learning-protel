from djitellopy import Tello
import cv2

# --- Drone Initialization ---
# Create a Tello object
tello = Tello()

# Connect to the drone

tello.connect()

# Turn on the video stream
tello.streamon()


# --- Main Loop ---
try:
    while True:
        # Get the current frame from the drone's camera
        frame = tello.get_frame_read().frame

        # The stream is 960x720. We can resize it for a smaller window if needed.
        # For example, to make it half the size:
        frame_resized = cv2.resize(frame, (480, 360))

        # Display the frame in a window
        cv2.imshow("Tello Camera Feed", frame_resized)

        # Wait for 1 millisecond, and check if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' key pressed. Stopping stream and landing...")
            break

finally:
    # --- Cleanup ---
    # Turn off the video stream
    tello.streamoff()

    # Destroy the OpenCV window
    cv2.destroyAllWindows()

    # Land the drone if it's flying (optional, for safety)
    # Note: The original code didn't have takeoff, but this is good practice.
    # If you are not taking off, you can comment out tello.land()
    # tello.land()

    print("Resources released. Program ended.")