import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

image = cv2.imread('Aishwariya_Rai_(face).jpg')
landmarks = get_landmarks(image)
image_with_landmarks = annotate_landmarks(image, landmarks)

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()
 
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Aishwariya_Rai')
cv2.imwrite('Aishwariya_Rai_(face).jpg', image)

axs[1].imshow(cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB))
axs[1].set_title('Aishwariya_Rai with Landmarks')
cv2.imwrite('image_with_landmarks.jpg', image)

plt.show()


