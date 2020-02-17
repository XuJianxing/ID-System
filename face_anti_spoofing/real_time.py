import face_recognition
import cv2
from sklearn.externals import joblib
import skimage.viewer
import skimage.feature
import skimage.color
import skimage.transform
import numpy as np


n = 8
r = 3
method = "uniform"
clf = joblib.load("./face_anti_spoofing/backup/svm_%s_%s_%s.plk" % (n, r, method))
# clf = joblib.load("8_3_uniform.plk")

def lbp(frame, face, n, r, method):
    try:
        x, y, w, h = face
        frame = frame[y: y + h, x: x + w]
        frame = skimage.transform.resize(frame, (64, 64), mode="constant")
        lbp = skimage.feature.local_binary_pattern(skimage.color.rgb2grey(frame), n, r, method)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= hist.sum()
        return hist
    except Exception as e:
        print(e)
        return None


# Get a reference to webcam #0 (the default one)
#video_capture = cv2.VideoCapture(0)

def face_anti(image):
    # Grab a single frame of video
    #ret, frame = video_capture.read()
    #name="6.PNG"
    # Resize frame of video to 1/4 size for faster face recognition processing
    #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_frame = cv2.resize(image, (0, 0), fx=1.0, fy=1.0)
    face_locations = face_recognition.face_locations(small_frame)
    print('face_locations', face_locations)
    result = "1"
    # Display the results
    for top, right, bottom, left in face_locations:
        # judge if the target is a person
        lbp_feature = lbp(small_frame, [left, top, right - left, bottom - top], n, r, method)
        if len(lbp_feature):
            result = clf.predict([lbp_feature])[0]
        else:
            result = 0

        #result = "person" if result else "photo"
        result = "1" if result else "0"
        '''
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, str(result), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        print("result")
        print(result)
        '''
    print("antispoof result", result)
    return result
    # Display the resulting image
    #cv2.imshow('Video', cv2.resize(frame, (0, 0), fx=1.5, fy=1.5))
    #cv2.waitKey()
    '''
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    '''

# Release handle to the webcam
#video_capture.release()
#cv2.destroyAllWindows()
