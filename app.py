from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO(r'models\yolov8\YOLOv8_Small_RDD.pt')

def detect_road_damage(frame):
    results = model(frame)
    annotated_frame = results[0].plot()  # Annotate the frame with detections
    return annotated_frame

def gen_frames():  
    camera = cv2.VideoCapture(0)  # Use 0 for webcam

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_road_damage(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)