from flask import Flask,render_template,Response
import cv2
from test import getASL

#running the project: $env:FLASK_APP = "app.py" $env:FLASK_DEVELOPMENT = "development" flask run
app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame = getASL(frame)

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

            

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)


# app = Flask(__name__, static_url_path='',
#                   static_folder='build',
#                   template_folder='build')

# @app.route("/")
# def hello():
#     return render_template("index.html")
