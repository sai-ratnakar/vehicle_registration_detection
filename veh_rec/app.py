from flask import Flask, Response
app = Flask(__name__)
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask_detect import number_plate_detect

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','mp4','MOV','webm','avi'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def home():
	return render_template('upload.html')

def gen_frames(vid): 
    pat = "C:/Users/prasa/OneDrive/Desktop/veh_rec/static/uploads/"+str(vid)
    camera = cv2.VideoCapture(pat)
    i=0
    print("===========>START<============")
    while True:
        i+=1
        success, frame = camera.read()  
        if not success:
            print("===========>DONE<============")
            break
        else:
            if i%3==0:
                frame = number_plate_detect(frame)
                  
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
    

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(gen_frames(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		return redirect(request.url)


if __name__ == "__main__":
    app.run(port=3000,debug=True)


    # set flask_env=development