from flask import Flask, render_template, request, redirect, session, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import os

app = Flask(__name__)
app.secret_key = "secret123"

# ======================
# DATABASE CONFIG
# ======================
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# ======================
# DATABASE MODELS
# ======================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))


class LoginLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    status = db.Column(db.String(20))


# CREATE DATABASE
with app.app_context():
    db.create_all()


# ======================
# FACE SETTINGS
# ======================
AUTHORIZED_ID = 1
CONFIDENCE_THRESHOLD = 50
face_matched = False

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists("trainer.yml"):
    recognizer.read("trainer.yml")


# ======================
# CAMERA STREAM
# ======================
def gen_frames():
    global face_matched
    camera = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:
                face = gray[y:y+h, x:x+w]

                try:
                    id, conf = recognizer.predict(face)

                    if id == AUTHORIZED_ID and conf < CONFIDENCE_THRESHOLD:
                        face_matched = True
                        print("Face matched!")
                except:
                    pass

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    finally:
        camera.release()


# ======================
# ROUTES
# ======================

# HOME
@app.route("/")
def home():
    return render_template("home.html")


# REGISTER
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        if User.query.filter_by(username=u).first():
            return "User already exists"

        new_user = User(username=u,password=p)
        db.session.add(new_user)
        db.session.commit()
        return redirect("/login")

    return render_template("register.html")


# LOGIN
@app.route("/login", methods=["GET","POST"])
def login():
    error=None

    if request.method=="POST":
        u=request.form["username"].strip()
        p=request.form["password"].strip()

        user = User.query.filter_by(username=u).first()

        if user and user.password == p:
            session["user"]=u

            log = LoginLog(username=u, status="success")
            db.session.add(log)
            db.session.commit()

            return redirect("/dashboard")

        else:
            log = LoginLog(username=u, status="failed")
            db.session.add(log)
            db.session.commit()

            error="Invalid username or password"

    return render_template("login.html",error=error)


# FACE LOGIN (ADMIN)
@app.route("/face")
def face():
    global face_matched
    face_matched=False
    return render_template("face.html")


@app.route("/video")
def video():
    return Response(gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/check_face")
def check_face():
    global face_matched

    if face_matched:
        session["user"]="Admin"
        return jsonify({"status":"success"})

    return jsonify({"status":"pending"})


# DASHBOARD
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")

    return render_template("dashboard.html",name=session["user"])


# ADMIN PANEL
@app.route("/admin")
def admin():

    if session.get("user") != "Admin":
        return redirect("/login")

    users = User.query.all()
    logs = LoginLog.query.order_by(LoginLog.id.desc()).limit(20)

    return render_template("admin.html", users=users, logs=logs)


# DELETE USER
@app.route("/delete_user/<int:id>")
def delete_user(id):

    if session.get("user") != "Admin":
        return redirect("/login")

    user = User.query.get(id)
    if user:
        db.session.delete(user)
        db.session.commit()

    return redirect("/admin")


# LOGOUT
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(debug=True)