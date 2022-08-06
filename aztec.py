import mediapipe as mp
import cv2
import numpy as np
import math
import time
import winsound
from tensorflow import keras
from playsound import playsound

y_value_list = []
x_value_list = [x for x in range(0, 50)]
y_max_min = [-50, 50]
width = 640
height = 480
interval = 0.1
graph_image = np.zeros((height,  width, 3), np.uint8)  # creates a black image
graph_image[:] = 0, 0, 0
x_point = 0
y_point = 0
point_time = 0
i = 0

ratioList = []
blinkCounter = 0
counter = 0


Draw_face = mp.solutions.drawing_utils
Mesh_of_Face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                               max_num_faces=1,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
Draw_face_Specs = Draw_face.DrawingSpec(thickness=1, circle_radius=2)


def plot_data_graph(j, y_value_list, x_value_list, y_max_min, w, h, interval, graph_image, x_point, y_point, point_time):
    if time.time() - point_time > interval:
        cv2.rectangle(graph_image, (0, 0), (w, h), (0, 200, 255), cv2.FILLED)
        cv2.line(graph_image, (0,  h // 2), (w,  h // 2), (0, 0, 0), 10)

        # for x in range(0,  w, 50):
        #     cv2.line( graph_image, (x, 0), (x,  h),
        #              (50, 50, 50), 1)
        #
        for y in range(0, h, 50):
            # cv2.line( graph_image, (0, y), ( w, y),
            #         (50, 50, 50), 1)
            cv2.putText(graph_image,
                        f'{int( y_max_min[1] - ((y / 50) * (( y_max_min[1] -  y_max_min[0]) / ( h / 50))))}',
                        (10, y), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 0), 1)

        y_point = int(np.interp(j, y_max_min, [0, h]))
        y_value_list.append(y_point)
        if len(y_value_list) == 50:
            y_value_list.pop(0)
        for i in range(0, len(y_value_list)):
            if i < 2:
                pass
            else:
                cv2.line(graph_image, (int((x_value_list[i - 1] * (w // 100))) - (w // 10),
                                       y_value_list[i - 1]),
                         (int((x_value_list[i] * (w // 100)) - (w // 10)),
                          y_value_list[i]), (0, 0, 0), 10)
        point_time = time.time()

    return graph_image


def findMesh_of_Face(img, draw=True):
    RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Mesh_of_Face.process(RGB_image)
    faces = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            if draw:
                Draw_face.draw_landmarks(img, faceLms, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                         Draw_face_Specs, Draw_face_Specs)
            face = []
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                face.append([x, y])
            faces.append(face)
    return img, faces


def findDistance(p1, p2, img=None):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    info = (x1, y1, x2, y2, cx, cy)
    if img is not None:
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return length, info, img
    else:
        return length


def seperate_image(temp, array):
    phb, pht, phl, phr = array[0], array[1], array[2], array[3]
    pelb, pelt, pell, pelr = array[4], array[5], array[6], array[7]
    perb, pert, perl, perr = array[8], array[9], array[10], array[11]
    face_crop = temp[pht[1]:phb[1], phl[0]:phr[0]]
    eye_left = temp[pelt[1]:pelb[1], pell[0]:pelr[0]]
    eye_right = temp[pert[1]:perb[1], perl[0]:perr[0]]
    return face_crop, eye_left, eye_right


def Distance(p1, p2, img=None):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    return length


def text_inside_rectangle(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
                          colorR=(0, 255, 0), font=cv2.FONT_HERSHEY_PLAIN,
                          offset=10, border=None, colorB=(0, 255, 0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)

    return img, [x1, y2, x2, y1]


def check(img):
    _, faces = findMesh_of_Face(img, draw=False)
    if faces:
        return True
    else:
        return False


def get_images(img):
    color = (0, 0, 255)
    temp = np.copy(img)
    black = 255 * np.zeros((200,  200, 3), np.uint8)
    depth_image = np.zeros((50,  250, 3), np.uint8)
    cv2.putText(black, "NO FACE", (100, 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    img, faces = findMesh_of_Face(img, draw=True)
    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        point_array = [face[152], face[10], face[234], face[454],
                       face[23], face[27], face[130], face[243],
                       face[253], face[257], face[463], face[359]]
        face_crop, eye_left, eye_right = seperate_image(temp, point_array)
        face_crop = cv2.resize(face_crop, (400, 500))
        eye_left = cv2.resize(eye_left, (150, 100))
        eye_right = cv2.resize(eye_right, (150, 100))

        eyeleftUp = face[159]
        eyeleftDown = face[23]
        eyeleftLeft = face[130]
        eyeleftRight = face[243]
        lenghtVer = Distance(eyeleftUp, eyeleftDown)
        lenghtHor = Distance(eyeleftLeft, eyeleftRight)
        global i
        i += 1
        if i == 100:
            i = 0
        ratio = -1 * int((lenghtVer / lenghtHor) * 100)
        global ratioList
        global blinkCounter
        global counter
        ratioList.append(ratio)
        ratioAvg = sum(ratioList) / len(ratioList)
        if len(ratioList) > 20:
            ratioList.pop(0)
        if abs(abs(ratio) - abs(ratioAvg)) > 5 and counter == 0:
            blinkCounter += 1
            color = (0, 255, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255, 0, 255)

        graph_plot = plot_data_graph(ratio, y_value_list, x_value_list, y_max_min,
                                     width, height, interval, graph_image, x_point, y_point, point_time)
        text_inside_rectangle(graph_plot, f'Blink Count: {blinkCounter}', (250, 50), scale=3,
                              colorR=color)
        w = findDistance(pointLeft, pointRight)
        W = 6.3
        # Finding distance
        f = 500
        d = (W * f) / w

        if(d > 30):
            text_inside_rectangle(depth_image, f'Depth: {int(d)}cm',
                                  (0, 30),
                                  scale=2)
            winsound.PlaySound(None, winsound.SND_ASYNC)
        else:
            text_inside_rectangle(depth_image, f'Depth: {int(d)}cm',
                                  (0, 30),
                                  scale=2, colorR=(0, 0, 255))
            winsound.PlaySound("assets/beep.wav",
                               winsound.SND_ASYNC | winsound.SND_ALIAS)

        return temp, img, face_crop, eye_left, eye_right, graph_plot, depth_image
    else:
        return temp, black, black, black, black, black, depth_image


def get_disease(img):
    disease_name = ['Uveitis', 'Normal', 'Glaucoma',
                    'Crossed_Eyes', 'Cataracts', 'Bulging_Eyes']
    new_model = keras.models.load_model('assets/my_model.h5')
    test_image = img
    test_image = cv2.resize(test_image, dsize=(
        150, 100), interpolation=cv2.INTER_CUBIC)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    my_array = [test_image]
    my_array = np.array(my_array)
    new_model.predict(my_array)
    pred = np.argmax(new_model.predict(my_array), axis=-1)
    return str(disease_name[pred[0]])


def main():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if success & check(frame):
            temp_tuple = get_images(frame)
            original_image, final_image, face_crop, eye_left, eye_right, blink_graph, depth_img = temp_tuple
            cv2.imshow('1', original_image)
            cv2.imshow('2', final_image)
            cv2.imshow('3', face_crop)
            cv2.imshow('4', eye_left)
            cv2.imshow('5', eye_right)
            cv2.imshow('6', blink_graph)
            cv2.imshow('7', depth_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
