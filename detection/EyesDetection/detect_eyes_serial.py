import os
import cv2
import numpy as np
import serial
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# 1-Chargement du modèle
model_path = os.path.join(os.getcwd(), "eyes_detector.model.h5")

print("[INFO] Chargement du modèle...")
model = tf.keras.models.load_model(model_path, compile=False)
print("[INFO] Modèle chargé avec succès !")

# 2-Configuration du port série
try:
    ser = serial.Serial("COM1", 9600, timeout=1)
    print("[INFO] Port série COM1 ouvert.")
except Exception as e:
    print("[ERREUR] Impossible d'ouvrir COM1 :", e)
    ser = None

# 3-Chargement Haar Cascade pour détecter les yeux
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# 4-Capture vidéo (webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERREUR] Impossible d'ouvrir la caméra.")
    exit()

print("[INFO] Détection démarrée...")

# 5-Boucle principale
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire une image.")
        break

    # Conversion en gris pour Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    eye_status = "eyes_open"  
    color = (0, 255, 0)

    for (x, y, w, h) in eyes:
        roi = frame[y:y+h, x:x+w]

        
        if len(roi.shape) == 2 or roi.shape[2] == 1:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        else:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Prétraitement pour MobileNetV2
        roi = cv2.resize(roi, (224, 224))
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        roi = np.expand_dims(roi, axis=0)

        # Prédiction
        prediction = model.predict(roi, verbose=0)[0]

        # prediction[0] -> eyes_closed, prediction[1] -> eyes_open
        if prediction[0] > prediction[1]:
            eye_status = "eyes_closed"
            color = (0, 0, 255)
        else:
            eye_status = "eyes_open"
            color = (0, 255, 0)

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, eye_status, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        break  # analyser le premier œil détecté

    # 6-Envoi au Raspberry via port série
    if ser is not None:
        try:
            ser.write(b'1' if eye_status == "eyes_closed" else b'0')
        except:
            print("[ERREUR] Impossible d'écrire sur le port série.")

    # 7-Affichage
    cv2.imshow("Eye Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 8-Nettoyage
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
print("[INFO] Programme terminé.")
