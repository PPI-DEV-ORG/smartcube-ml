import cv2
from ultralytics import YOLO
from ultralytics.solutions import distance_calculation
import numpy as np
import torch

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the YOLOv8 model
model = YOLO('yolov8n.pt').to(device=device)
names = model.model.names

# Open the video file
# video_path = "/dev/video0"
video_path = "/home/ilham/Yolo/The CCTV People Demo 2.mp4"
cap = cv2.VideoCapture(video_path)

# Declare alertFlag
alerted_items = []

# Declare lines points array
linesPoints = []

# Declare lines points draw function
def draw_line_between_points(image, points):
    if len(points) >= 2:
        cv2.line(image, (int(points[-1][0]),int(points[-1][1])), (int(points[-2][0]),int(points[-2][1])), (0, 255, 0), 10)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Declare array to store object
        objectStorage = []

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Get frame dimensions
        height, width, _ = annotated_frame.shape

        # Define yMax, yMin, xMax, xMin of the area
        xMaxTheArea = width // 3
        xMinTheArea = 0
        yMaxTheArea = height
        yMinTheArea = 0

        # Draw the red and green regions (picture, left top point, right bottom point, color, weight)
        cv2.rectangle(annotated_frame, (xMinTheArea, yMinTheArea), (xMaxTheArea, yMaxTheArea), (0, 255, 0), 5)  # Green region

        # Check if result is not None
        if results is not None and len(results) > 0:
            # Catch all detected object
            boxes = results[0].boxes
            for box in boxes:
                # If Human detect, set the state to True
                if box.cls[0] == 0:
                    if box.id != None:
                        # Set flag
                        itemExist = False

                        # Do looping to check if item exist
                        for item in objectStorage:
                            if item["id"] == box.id[0] and item["class"] == box.cls[0]:
                                itemExist = True
                        
                        # Add item if not exist
                        if itemExist == False:
                            objectStorage.append({
                                "id": box.id[0],
                                "class": box.cls[0],
                                "xMax": box.xyxy[0][0],
                                "yMax": box.xyxy[0][1],
                                "xMin": box.xyxy[0][2],
                                "yMin": box.xyxy[0][3]
                            })

        # Check if someone in the area
        for object in objectStorage:
            if object["xMax"] >= xMinTheArea and object["xMax"] <= xMaxTheArea and object["xMin"] >= xMinTheArea and object["xMin"] <= xMaxTheArea and object["yMax"] >= yMinTheArea  and object["yMax"] <= yMaxTheArea and object["yMin"] >= yMinTheArea  and object["yMin"] <= yMaxTheArea :
                
                # Get bottom-middle point of bounding box (tracks come from person foot)
                xMid = (object["xMax"] - object["xMin"]) / 2
                yMid = object["yMin"] # Assume as her/his foot
                
                # Set flag to check object points
                linesPointsFlag = False

                # Do looping to to check if object points is stored or not
                for line in linesPoints:
                    # If found
                    if line["class"] == object["class"] and line["id"] == object["id"]:
                        # Set flag to true
                        linesPointsFlag = True

                        if len(line["points"]) <= 20:
                            # Add new point
                            line["points"].append([xMid,yMid])
                        else:
                            # remove and restore index queue
                            line["points"].pop(0)
                            # Add new point
                            line["points"].append([xMid,yMid])
                        
                # If not found
                if linesPointsFlag == False:
                    linesPoints.append({'class':object["class"], 'id':object["id"], 'points':[]})

                # Check if the item has not been alerted already
                alertFound = False
                for alert_key in alerted_items:
                    if(alert_key["id"] == object["id"] and alert_key["class"] == object["class"]):
                        alertFound = True

                if alertFound == False:
                    # Create a unique identifier for the alerted item
                    alerted_items.append({"id" : object["id"], "class" : object["class"]})

                    # Print a specific alert message
                    print(f'A human with class : {object["class"]} and id : {object["id"]}, has entered the area !!')

                # Draw the human
                cv2.rectangle(annotated_frame,
                    (int(object["xMin"]), int(object["yMin"])),
                    (int(object["xMax"]), int(object["yMax"])),
                    (200, 250, 0), 5)

        # Draw the line
        for line in linesPoints:
            if len(line["points"]) >= 2:
                cv2.line(annotated_frame, (int(line["points"][-1][0]),int(line["points"][-1][1])), (int(line["points"][-2][0]),int(line["points"][-2][1])), (0, 255, 0), 10)

        # Display the frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
