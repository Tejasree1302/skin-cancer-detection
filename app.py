import os
import numpy as np
from flask import Flask, render_template, request, redirect, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from create_database import setup_database
from utils import login_required, set_session
from argon2 import PasswordHasher
import sqlite3, contextlib, re
import gdown   # NEW

# Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Database
database = "users.db"
setup_database(database)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==============================
# 🔥 MODEL LOADING (UPDATED)
# ==============================

model_path = "model/model_resnet.h5"

# Auto download if model not present
if not os.path.exists(model_path):
    print("📥 Model not found. Downloading...")

    # 🔁 Replace this with your actual Google Drive file ID
    url = "https://drive.google.com/file/d/1jxyRRPfW1BijJlW-q6Ntz3Xk5WnFUuoe/view?usp=sharing"

    os.makedirs("model", exist_ok=True)
    gdown.download(url, model_path, quiet=False)

# Load trained model
model = load_model(model_path)

# Classes
class_names = ['Basal Cell Carcinoma', 'Melanoma', 'Squamous Cell Carcinoma']
CONFIDENCE_THRESHOLD = 0.6

# ==============================
# ROUTES
# ==============================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    username = request.form.get('username')
    password = request.form.get('password')
    
    query = 'SELECT username, password, email FROM users WHERE username=:username'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        account = conn.execute(query, {'username': username}).fetchone()
    
    if not account:
        return render_template('login.html', error='Username does not exist')

    try:
        ph = PasswordHasher()
        ph.verify(account[1], password)
    except:
        return render_template('login.html', error='Incorrect password')

    set_session(username=account[0], email=account[2])
    return redirect('/predict_page')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='GET':
        return render_template('register.html')
    
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm = request.form.get('confirm-password')

    if len(password)<8:
        return render_template('register.html', error='Password must be >=8 chars')
    if password != confirm:
        return render_template('register.html', error='Passwords do not match')
    if not re.match(r'^[a-zA-Z0-9]+$', username):
        return render_template('register.html', error='Username invalid')

    ph = PasswordHasher()
    hashed_pw = ph.hash(password)

    query = 'INSERT INTO users(username,password,email) VALUES(:username,:password,:email)'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        conn.execute(query, {'username':username, 'password':hashed_pw, 'email':email})

    set_session(username, email)
    return redirect('/')

@app.route('/predict_page')
@login_required
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        return render_template('predict.html', error="No image uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('predict.html', error="No file selected")
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        img = load_img(file_path, target_size=(224,224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        confidence_scores = predictions[0]

        if max(confidence_scores) < CONFIDENCE_THRESHOLD:
            predicted_class = "Non-Cancerous"
        else:
            predicted_class = class_names[np.argmax(confidence_scores)]

        confidence = max(confidence_scores)*100
        return render_template('result.html', image=file.filename,
                               predicted_class=predicted_class,
                               confidence=f"{confidence:.2f}%")
    except Exception as e:
        return render_template('predict.html', error=str(e))

if __name__=="__main__":
    app.run(debug=True)