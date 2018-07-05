import cv2 
import caffe

def _load_model(prototxt,caffemodel):
    net = caffe.Net(prototxt,      # defines the structure of the model
                    caffemodel,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

def opencv_detect(image):
    cascPath = 'haarcascade_frontalface_alt2.xml'
    facesCascade = cv2.CascadeClassifier(cascPath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = facesCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    return faces

if __name__ == '__main__':
    path = 'sample.png'
    prototxt = 'yolo_v3.prototxt'
    caffemodel = 'gnet_yolo_v3.caffemodel'
    #_load_model(prototxt,caffemodel)

    image = cv2.imread(path)
    faces = opencv_detect(image)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    print(faces)
    cv2.imshow('faces found', image)
    cv2.waitKey(0)
