from ultralytics import YOLO
import cv2
import dlib
import time
from datetime import datetime
# import os
import numpy as np
import easyocr
from sort.sort import Sort
import util
# import csv

results = {}

# Initialize cascade classifier for detecting cars
carCascade = cv2.CascadeClassifier('files/HaarCascadeClassifier.xml')

# Initialize the EasyOCR Reader object for license plate recognition
reader = easyocr.Reader(['en'])

# Initialize the YOLO model for vehicle detection
vehicles = [2, 3, 5, 7]
coco_model = YOLO('yolov8n.pt')

# Initialize the YOLO model for license plate detection
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Initialize the SORT tracker for vehicle tracking
mot_tracker = Sort()

# Constants for overspeeding detection
WIDTH = 1280
HEIGHT = 720
cropBegin = 240
mark1 = 120
mark2 = 360
markGap = 15
fpsFactor = 3
speedLimit = 20
startTracker = {}
endTracker = {}

def blackout(image):
    xBlack = 360
    yBlack = 300
    triangle_cnt = np.array( [[0,0], [xBlack,0], [0,yBlack]] )
    triangle_cnt2 = np.array( [[WIDTH,0], [WIDTH-xBlack,0], [WIDTH,yBlack]] )
    cv2.drawContours(image, [triangle_cnt], 0, (0,0,0), -1)
    cv2.drawContours(image, [triangle_cnt2], 0, (0,0,0), -1)

    return image

# Function to save car image, date, time, speed
def saveCar(speed, image):
    now = datetime.today().now()
    nameCurTime = now.strftime("%d-%m-%Y-%H-%M-%S-%f")
    link = 'overspeeding/cars/' + nameCurTime + '.jpeg'
    cv2.imwrite(link, image)

# Function to calculate speed
def estimateSpeed(carID):
    timeDiff = endTracker[carID] - startTracker[carID]
    speed = round(markGap / timeDiff * fpsFactor * 3.6, 2)
    return speed

# Function to track cars
def trackMultipleObjects(video):
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    carTracker = {}

    while True:
        rc, image = video.read()
        if type(image) == type(None):
            break

        frameTime = time.time()
        image = cv2.resize(image, (WIDTH, HEIGHT))[cropBegin:720, 0:1280]
        resultImage = blackout(image)
        cv2.line(resultImage, (0, mark1), (1280, mark1), (0, 0, 255), 2)
        cv2.line(resultImage, (0, mark2), (1280, mark2), (0, 0, 255), 2)

        frameCounter = frameCounter + 1

        # Delete carIDs not in frame
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)

        # Main program
        if frameCounter % 60 == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))  # Detect cars in frame

            for (_x, _y, _w, _h) in cars:
                # Get position of a car
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                xbar = x + 0.5 * w
                ybar = y + 0.5 * h

                matchCarID = None

                # If centroid of current car is near the centroid of another car in previous frame, they are the same
                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    tx = int(trackedPosition.left())
                    ty = int(trackedPosition.top())
                    tw = int(trackedPosition.width())
                    th = int(trackedPosition.height())

                    txbar = tx + 0.5 * tw
                    tybar = ty + 0.5 * th

                    if (
                        (tx <= xbar <= (tx + tw))
                        and (ty <= ybar <= (ty + th))
                        and (x <= txbar <= (x + w))
                        and (y <= tybar <= (y + h))
                    ):
                        matchCarID = carID

                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker

                    currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            tx = int(trackedPosition.left())
            ty = int(trackedPosition.top())
            tw = int(trackedPosition.width())
            th = int(trackedPosition.height())

            # Put bounding boxes
            cv2.rectangle(resultImage, (tx, ty), (tx + tw, ty + th), rectangleColor, 2)
            cv2.putText(
                resultImage,
                str(carID),
                (tx, ty - 5),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
                1,
            )

            # Estimate speed
            if (
                carID not in startTracker
                and mark2 > ty + th > mark1
                and ty < mark1
            ):
                startTracker[carID] = frameTime

            elif carID in startTracker and carID not in endTracker and mark2 < ty + th:
                endTracker[carID] = frameTime
                speed = estimateSpeed(carID)
                if speed > speedLimit:
                    print(f"CAR-ID : {carID} : {speed} kmph - OVERSPEED")
                    saveCar(speed, image[ty : ty + th, tx : tx + tw])
                    # Detect license plate for overspeeding car
                    license_plates = license_plate_detector(image)[0]
                    for license_plate in license_plates.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = license_plate

                        # Assign license plate to car
                        xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(license_plate, carID)

                        if car_id == carID:
                            print(f"License plate detected for overspeeding car ID {car_id}")

                            # Crop license plate
                            license_plate_crop = image[int(y1) : int(y2), int(x1) : int(x2), :]

                            # Process license plate
                            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                            # Read license plate number
                            license_plate_text, license_plate_text_score = util.read_license_plate(license_plate_crop_thresh)

                            if license_plate_text is not None:
                                results[frameCounter][car_id]["overspeeding_car"] = {
                                    "license_plate": {
                                        "bbox": [x1, y1, x2, y2],
                                        "text": license_plate_text,
                                        "bbox_score": score,
                                        "text_score": license_plate_text_score,
                                    }
                                }
                                print(f"License plate text: {license_plate_text}")

        # Display each frame
        cv2.imshow("result", resultImage)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
    
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load video
    video = cv2.VideoCapture("./files/sample.mp4")

    # Run the object tracking and speed detection
    trackMultipleObjects(video)