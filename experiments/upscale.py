import cv2 as cv
import onnxruntime
import numpy as np


session = onnxruntime.InferenceSession(
    'models/4xNomos2_hq_dat2_fp32.onnx',
    providers=['CPUExecutionProvider']
)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


def upscale(img):
    # this still has clipping issues?
    img = img.transpose(2, 0, 1) / 255.0

    res = session.run([], {'input': [img]})[0][0]

    res = res.transpose(1, 2, 0) * 255.0
    res = res.astype(np.uint8)
    return res


def main():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = cv.resize(frame, (128, 128))
        img = upscale(img)
        # Display the resulting frame
        cv.imshow('frame', img)
        if cv.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
