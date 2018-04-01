import cv2
eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
face_cascPath = 'haarcascade_frontalface_alt.xml'  #face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)


class Eye_Detect():
    def __init__(self, cam_flag=0):
        self.cap = cv2.VideoCapture(cam_flag)
    def get_frames(self):
        ret, img = self.cap.read()
        return ret, img
    def detect(self):
        # ret, img = self.cap.read()
        ret, img = self.get_frames()
        eye_flag = 253
        if ret:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            # print("Found {0} faces!".format(len(faces)))
            if len(faces) > 0:
                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
                frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
                eyes = eyeCascade.detectMultiScale(
                    frame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    # flags = cv2.CV_HAAR_SCALE_IMAGE
                )
                if len(eyes) == 0:
                    eye_flag = 255
                    print('no eyes!!!')
                else:
                    eye_flag = 254
                    print('eyes!!!')
                frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Face Recognition', frame_tmp)
            else:
                eye_flag = 253
                print('no faces!!!')
            waitkey = cv2.waitKey(1)
            if waitkey == ord('q') or waitkey == ord('Q'):
                cv2.destroyAllWindows()
                    # break
        return eye_flag
