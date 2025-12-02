from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

#PARAMÈTRES
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Chemins des dossiers
TRAIN_DIR = r"C:\Users\zeine\OneDrive\Desktop\projet_iot\detection\EyesDetection\dataset\train"
TEST_DIR = r"C:\Users\zeine\OneDrive\Desktop\projet_iot\detection\EyesDetection\dataset\test"

print(f"[INFO] Dossier train: {TRAIN_DIR}")
print(f"[INFO] Dossier test: {TEST_DIR}")

#CHARGEMENT DES IMAGES
print("[INFO] Chargement des images d'yeux...")

def load_images_from_folder(folder_path):
    """Charge les images depuis un dossier avec sous-dossiers"""
    data = []
    labels = []

    if not os.path.exists(folder_path):
        print(f"[ERREUR] Le dossier {folder_path} n'existe pas!")
        return data, labels

    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        if not os.path.isdir(category_path):
            continue

        print(f"[INFO] Traitement catégorie: {category}")
        image_count = 0

        for img_file in os.listdir(category_path):
            try:
                img_path = os.path.join(category_path, img_file)
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    continue

                image = load_img(img_path, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                data.append(image)

                # Normalisation des labels
                if 'open' in category.lower():
                    labels.append('eyes_open')
                elif 'close' in category.lower() or 'closed' in category.lower():
                    labels.append('eyes_closed')
                else:
                    labels.append(category)

                image_count += 1
            except Exception as e:
                print(f"[ERREUR] Impossible de charger {img_path}: {e}")

        print(f"[INFO] {image_count} images chargées pour {category}")

    return data, labels

# Charger les données
train_data, train_labels = load_images_from_folder(TRAIN_DIR)
test_data, test_labels = load_images_from_folder(TEST_DIR)

# Combiner pour le split
all_data = train_data + test_data
all_labels = train_labels + test_labels

print(f"[INFO] TOTAL: {len(all_data)} images chargées")

if len(all_data) == 0:
    print("[ERREUR] Aucune image trouvée! Vérifiez vos dossiers.")
    exit()

#PRÉPARATION DES DONNÉES
lb = LabelBinarizer()
labels_encoded = lb.fit_transform(all_labels)
labels_encoded = to_categorical(labels_encoded)

data_array = np.array(all_data, dtype="float32")
labels_array = np.array(labels_encoded)

print(f"[INFO] Classes détectées: {lb.classes_}")

# Division train/test
(trainX, testX, trainY, testY) = train_test_split(
    data_array, labels_array,
    test_size=0.20,
    stratify=labels_array,
    random_state=42
)

print(f"[INFO] Ensemble d'entraînement: {len(trainX)} images")
print(f"[INFO] Ensemble de test: {len(testX)} images")

#AUGMENTATION DES DONNÉES
aug = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.10,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.10,
    horizontal_flip=True,
    fill_mode="nearest"
)

#CONSTRUCTION DU MODÈLE
print("[INFO] Construction du modèle de détection d'yeux...")

baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Geler les couches de base
for layer in baseModel.layers:
    layer.trainable = False

#COMPILATION
print("[INFO] Compilation du modèle...")
opt = Adam(learning_rate=INIT_LR)  
model.compile(
    loss="binary_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

#CALLBACK POUR LR
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    verbose=1,
    min_lr=1e-7
)

#ENTRAÎNEMENT
print("[INFO] Entraînement du modèle d'yeux...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
    callbacks=[reduce_lr]
)

#ÉVALUATION
print("[INFO] Évaluation du réseau...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(
    testY.argmax(axis=1),
    predIdxs,
    target_names=lb.classes_
))

# Sauvegarde du modèle
model_path = r"C:\Users\zeine\OneDrive\Desktop\projet_iot\detection\EyesDetection\eyes_detector.model.h5"
print(f"[INFO] Sauvegarde du modèle : {model_path}")
model.save(model_path)
print("[INFO] Modèle sauvegardé avec succès!")


#GRAPHIQUE
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Entraînement Détection Yeux")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

plot_path = r"C:\Users\zeine\OneDrive\Desktop\projet_iot\detection\EyesDetection\eyes_training_plot.png"
plt.savefig(plot_path)
print(f"[INFO] Graphique sauvegardé : {plot_path}")
