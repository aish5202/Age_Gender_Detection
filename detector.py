import cv2
import numpy as np
import time
from datetime import datetime


# ====================================================
# MODEL PATHS
# ====================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models/")

FACE_PROTO = MODEL_PATH + "opencv_face_detector.pbtxt"
FACE_MODEL = MODEL_PATH + "opencv_face_detector_uint8.pb"

AGE_PROTO = MODEL_PATH + "age_deploy.prototxt"
AGE_MODEL = MODEL_PATH + "age_net.caffemodel"

GENDER_PROTO = MODEL_PATH + "gender_deploy.prototxt"
GENDER_MODEL = MODEL_PATH + "gender_net.caffemodel"


# ====================================================
# LOAD MODELS
# ====================================================

print("Loading models...")

faceNet = cv2.dnn.readNet(
    FACE_MODEL,
    FACE_PROTO
)

ageNet = cv2.dnn.readNet(
    AGE_MODEL,
    AGE_PROTO
)

genderNet = cv2.dnn.readNet(
    GENDER_MODEL,
    GENDER_PROTO
)

print("Models loaded successfully")


# ====================================================
# CONSTANTS
# ====================================================

MODEL_MEAN_VALUES = (
    78.4263377603,
    87.7689143744,
    114.895847746
)


AGE_LIST = [
    "0-2",
    "4-6",
    "8-12",
    "15-20",
    "25-32",
    "38-43",
    "48-53",
    "60-100"
]


GENDER_LIST = [
    "Male",
    "Female"
]


FACE_CONFIDENCE_THRESHOLD = 0.7



# ====================================================
# IMAGE PREPROCESSING
# ====================================================

def preprocess_image(image):

    if image is None:
        raise ValueError("Input image is None")


    # If numpy image
    if isinstance(image, np.ndarray):

        frame = image.copy()

    else:

        frame = np.array(image)

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )


    h,w = frame.shape[:2]


    max_width = 900


    if w > max_width:

        scale = max_width / w

        frame = cv2.resize(
            frame,
            (
                int(w*scale),
                int(h*scale)
            )
        )


    frame = cv2.GaussianBlur(
        frame,
        (3,3),
        0
    )


    return frame



# ====================================================
# FACE DETECTION
# ====================================================

def detect_faces(frame):


    if frame is None:
        return []


    h,w = frame.shape[:2]


    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300,300),
        (104,177,123)
    )


    faceNet.setInput(blob)

    detections = faceNet.forward()


    faces=[]


    for i in range(
        detections.shape[2]
    ):


        confidence = detections[0,0,i,2]


        if confidence > FACE_CONFIDENCE_THRESHOLD:


            box = detections[0,0,i,3:7] * np.array(
                [w,h,w,h]
            )


            x1,y1,x2,y2 = box.astype(int)


            faces.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    float(confidence)
                )
            )


    return faces



# ====================================================
# FACE PREPROCESSING
# ====================================================

def preprocess_face(frame,face):


    h,w = frame.shape[:2]


    x1,y1,x2,y2,confidence = face


    padding = 20


    x1=max(0,x1-padding)
    y1=max(0,y1-padding)

    x2=min(w,x2+padding)
    y2=min(h,y2+padding)



    face_img = frame[
        y1:y2,
        x1:x2
    ]


    if face_img.size == 0:
        return None,None



    face_img=cv2.resize(
        face_img,
        (227,227)
    )



    blob=cv2.dnn.blobFromImage(
        face_img,
        1,
        (227,227),
        MODEL_MEAN_VALUES,
        swapRB=False
    )


    return blob,(x1,y1,x2,y2)




# ====================================================
# PREDICTIONS
# ====================================================


def predict_gender(blob):

    genderNet.setInput(blob)

    pred=genderNet.forward().flatten()

    idx=np.argmax(pred)


    return (
        GENDER_LIST[idx],
        float(pred[idx])
    )




def predict_age(blob):

    ageNet.setInput(blob)

    pred=ageNet.forward().flatten()

    idx=np.argmax(pred)


    return (
        AGE_LIST[idx],
        float(pred[idx])
    )





# ====================================================
# DRAW RESULTS
# ====================================================

def draw_result(
        frame,
        coords,
        gender,
        gender_conf,
        age,
        face_conf
):


    x1,y1,x2,y2=coords


    color=(0,255,0) if gender=="Male" else (255,0,255)


    cv2.rectangle(
        frame,
        (x1,y1),
        (x2,y2),
        color,
        2
    )


    text=f"{gender} {gender_conf*100:.1f}% | Age {age}"


    cv2.putText(
        frame,
        text,
        (x1,max(20,y1-10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )



# ====================================================
# MAIN FUNCTION
# ====================================================


def process_image(image):


    start=time.time()


    frame=preprocess_image(image)


    faces=detect_faces(frame)


    results=[]


    male=0
    female=0



    for face in faces:


        blob,coords=preprocess_face(
            frame,
            face
        )


        if blob is None:
            continue



        gender,g_conf=predict_gender(blob)

        age,a_conf=predict_age(blob)



        if gender=="Male":
            male+=1
        else:
            female+=1



        draw_result(
            frame,
            coords,
            gender,
            g_conf,
            age,
            face[4]
        )



        results.append(
            {
                "gender":gender,
                "gender_conf":round(g_conf,4),
                "age":age,
                "age_conf":round(a_conf,4),
                "face_conf":round(face[4],4)
            }
        )



    processing_time=round(
        time.time()-start,
        3
    )


    timestamp=datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )


    cv2.putText(
        frame,
        f"Faces: {len(results)}",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )


    return (
        frame,
        results,
        processing_time,
        timestamp
    )