from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

TF_ENABLE_ONEDNN_OPTS=0
from keras.models import load_model
from keras.preprocessing import image
 
from keras.applications.vgg19 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import tempfile
import os



app = FastAPI(
    title="DrQA backend API", docs_url="/docs"
)


@app.get("/")
async def root(request: Request):
    return {"message": "Server is up and running!"}


model = load_model('models/1')
# print('Model loaded')


def predict_single_image(model, img_path):
    
    class_names = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala']
    # Load and preprocess the image
    img_height = 224
    img_width = 224
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Make predictions
    predictions = model.predict(img_array, verbose=0)

    # Get predicted class and confidence
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return predicted_class, confidence



@app.post("/predict")
async def predict_image(image: UploadFile):
    # Create temporary directory for uploaded image
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded image to temporary file
        img_path = f"{temp_dir}/temp.jpg"
        with open(img_path, "wb") as f:
            f.write(await image.read())

        # Predict class and confidence for the saved image
        predicted_class, confidence = predict_single_image(model, img_path)

        # Delete temporary image and directory
        os.remove(img_path)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=8000)
