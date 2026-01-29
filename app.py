from flask import Flask, request, jsonify, render_template
import speech_recognition as sr
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/audio")
def audio():
    return render_template("audio.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    
    return jsonify({"success": True, "text": text})

if __name__ == "__main__":
    app.run(debug=True)