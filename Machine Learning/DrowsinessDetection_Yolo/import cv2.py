import time
import cv2

def save_image(frame, filename):
  """Saves the captured frame to a file.

  Args:
      frame: The captured image frame from the camera.
      filename: The desired filename to save the image.
  """
  cv2.imwrite(filename, frame)
  print(f"Image saved as: {filename}")

def set_resolution(cap, width, height):
  """Sets the camera resolution.

  Args:
      cap: The VideoCapture object representing the camera.
      width: The desired width of the frame.
      height: The desired height of the frame.
  """
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  print(f"Resolution set to: {width}x{height}")

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Set camera resolution (optional)
# Uncomment these lines to set a specific resolution
# set_resolution(cap, 640, 480)  # Example resolution

# Check if camera opened successfully
if not cap.isOpened():
  print("Error opening camera")
  exit()

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Check if frame is read correctly
  if not ret:
    print("Failed to capture frame")
    break

  # Display the resulting frame
  cv2.imshow('Camera', frame)

  # Press 's' to save the frame with timestamp
  if cv2.waitKey(1) == ord('s'):
    filename = f"captured_image_{int(time.time())}.png"  # Add timestamp
    save_image(frame, filename)

  # Press 'r' to set camera resolution
  if cv2.waitKey(1) == ord('r'):
    width = int(input("Enter desired width: "))
    height = int(input("Enter desired height: "))
    set_resolution(cap, width, height)

  # Press 'q' to exit the program
  if cv2.waitKey(1) == ord('q'):
    break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
