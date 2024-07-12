import cv2
import os
import time

def capture_images(num_images, folder_path):
  """
  Captures specified number of images from webcam and saves them to the specified folder.

  Args:
    num_images: Number of images to capture.
    folder_path: Path to the folder where images will be saved.
  """

  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  cap = cv2.VideoCapture(0)  # Access the default camera (index 0)

  for i in range(num_images):
    ret, frame = cap.read()

    if not ret:
      print("Error capturing frame")
      break
      
      # Introduce a 500 millisecond delay
    time.sleep(0.5)

    image_path = os.path.join(folder_path, f"image_{i+1}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Image {i+1} saved to {image_path}")

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  num_images = 20  # Number of images to capture
  folder_path = "D:\Projects\DataScience\Machine Learning\DrowsinessDetection_Yolo\image_glass\drowsy"  # Folder to save images

  capture_images(num_images, folder_path)
