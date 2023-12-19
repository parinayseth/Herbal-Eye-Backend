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
    plant_cure = {
    'Aloevera': ['Burns', 'Digestive Issues', 'Skin Conditions', 'Wound Healing', 'Inflammation'],
    'Amla': ['Digestive Problems', 'Heart Health', 'Hair and Skin Conditions', 'Liver Health', 'Eye Health'],
    'Amruthaballi': ['Respiratory Issues', 'Anti-inflammatory', 'Stress Relief', 'Boosts Immunity', 'Anti-bacterial'],
    'Arali': ['Anti-inflammatory', 'Stress Relief', 'Boosts Energy', 'Respiratory Conditions', 'Joint Pain'],
    'Astma_weed': ['Asthma', 'Respiratory Conditions', 'Anti-inflammatory', 'Bronchitis', 'Cough'],
    'Badipala': ['Anti-inflammatory', 'Detoxification', 'Digestive Issues', 'Stress Relief', 'Liver Health'],
    'Balloon_Vine': ['Skin Conditions', 'Detoxification', 'Anti-inflammatory', 'Stress Relief', 'Digestive Issues'],
    'Bamboo': ['Anti-inflammatory', 'Digestive Health', 'Joint Pain', 'Detoxification', 'Bone Health'],
    'Beans': ['Weight Management', 'Diabetes Management', 'Digestive Health', 'Heart Health', 'Anti-inflammatory'],
    'Betel': ['Oral Health', 'Stress Relief', 'Digestive Issues', 'Anti-bacterial', 'Cough'],
    'Bhrami': ['Memory Enhancement', 'Stress Relief', 'Depression', 'Cognitive Health', 'Anxiety'],
    'Bringaraja': ['Hair and Scalp Health', 'Digestive Health', 'Detoxification', 'Anti-inflammatory', 'Liver Health'],
    'Caricature': ['Anti-inflammatory', 'Joint Pain', 'Detoxification', 'Skin Conditions', 'Digestive Health'],
    'Castor': ['Laxative', 'Skin Conditions', 'Digestive Issues', 'Joint Pain', 'Anti-inflammatory'],
    'Catharanthus': ['Blood Pressure Regulation', 'Blood Disorders', 'Diabetes Management', 'Anti-cancer', 'Antimicrobial'],
    'Chakte': ['Digestive Issues', 'Detoxification', 'Anti-inflammatory', 'Respiratory Conditions', 'Liver Health'],
    'Chilly': ['Anti-inflammatory', 'Metabolism Boost', 'Pain Relief', 'Digestive Health', 'Weight Management'],
    'Citron_lime': ['Digestive Health', 'Vitamin C Boost', 'Immune Support', 'Anti-inflammatory', 'Skin Conditions'],
    'Coffee': ['Antioxidant', 'Liver Health', 'Mental Alertness', 'Metabolism Boost', 'Heart Health'],
    'Common_rue': ['Joint Pain', 'Skin Conditions', 'Digestive Issues', 'Anti-inflammatory', 'Respiratory Conditions'],
    'Coriander': ['Cholesterol Regulation', 'Antioxidant', 'Anti-inflammatory', 'Digestive Health', 'Blood Sugar Regulation'],
    'Curry': ['Digestive Health', 'Heart Health', 'Anti-inflammatory', 'Joint Pain', 'Liver Health'],
    'Doddpathre': ['Joint Pain', 'Anti-inflammatory', 'Digestive Issues', 'Detoxification', 'Respiratory Conditions'],
    'Drumstick': ['Nutrient Boost', 'Blood Sugar Regulation', 'Digestive Health', 'Heart Health', 'Anti-inflammatory'],
    'Ekka': ['Anti-inflammatory', 'Pain Relief', 'Joint Pain', 'Respiratory Conditions', 'Skin Conditions'],
    'Eucalyptus': ['Pain Relief', 'Anti-inflammatory', 'Cough Relief', 'Respiratory Health', 'Immune Support'],
    'Ganigale': ['Digestive Health', 'Detoxification', 'Skin Conditions', 'Anti-inflammatory', 'Joint Pain'],
    'Ganike': ['Digestive Health', 'Detoxification', 'Anti-inflammatory', 'Joint Pain', 'Skin Conditions'],
    'Gasagase': ['Heart Health', 'Digestive Health', 'Anti-inflammatory', 'Joint Pain', 'Skin Conditions'],
    'Ginger': ['Digestive Health', 'Anti-inflammatory', 'Immune Support', 'Nausea Relief', 'Joint Pain'],
    'Globe_Amarnath': ['Anti-inflammatory', 'Digestive Health', 'Detoxification', 'Joint Pain', 'Heart Health'],
    'Guava': ['Anti-inflammatory', 'Heart Health', 'Digestive Health', 'Immune Support', 'Skin Conditions'],
    'Henna': ['Cooling Effect', 'Hair Health', 'Wound Healing', 'Skin Conditions', 'Anti-inflammatory'],
    'Hibiscus': ['Hair Health', 'Digestive Health', 'Blood Pressure Regulation', 'Liver Health', 'Anti-inflammatory'],
    'Honge': ['Joint Pain', 'Skin Conditions', 'Anti-inflammatory', 'Respiratory Conditions', 'Digestive Health'],
    'Insulin': ['Heart Health', 'Digestive Health', 'Anti-inflammatory', 'Diabetes Management', 'Blood Sugar Regulation'],
    'Jackfruit': ['Skin Conditions', 'Immune Support', 'Heart Health', 'Digestive Health', 'Anti-inflammatory'],
    'Jasmine': ['Anxiety', 'Stress Relief', 'Anti-inflammatory', 'Sleep Aid', 'Skin Conditions'],
    'Kambajala': ['Anti-inflammatory', 'Digestive Health', 'Joint Pain', 'Skin Conditions', 'Respiratory Conditions'],
    'Kasambruga': ['Anti-inflammatory', 'Hair Health', 'Digestive Health', 'Detoxification', 'Liver Health'],
    'Kohlrabi': ['Heart Health', 'Digestive Health', 'Anti-inflammatory', 'Skin Conditions', 'Detoxification'],
    'Lantana': ['Detoxification', 'Digestive Health', 'Anti-inflammatory', 'Joint Pain', 'Respiratory Conditions'],
        'Lemon': ['Skin Conditions', 'Digestive Health', 'Detoxification', 'Immune Support', 'Anti-inflammatory'],
    'Lemongrass': ['Digestive Health', 'Anti-inflammatory', 'Stress Relief', 'Respiratory Conditions', 'Detoxification'],
    'Malabar_Nut': ['Asthma', 'Respiratory Conditions', 'Anti-inflammatory', 'Cough Relief', 'Bronchitis'],
    'Malabar_Spinach': ['Anti-inflammatory', 'Bone Health', 'Eye Health', 'Digestive Health', 'Heart Health'],
    'Mango': ['Digestive Health', 'Immune Support', 'Skin Conditions', 'Anti-inflammatory', 'Heart Health'],
    'Marigold': ['Digestive Health', 'Vision Health', 'Anti-inflammatory', 'Wound Healing', 'Skin Conditions'],
    'Mint': ['Digestive Health', 'Stress Relief', 'Anti-inflammatory', 'Respiratory Conditions', 'Headache Relief'],
    'Neem': ['Anti-bacterial', 'Skin Conditions', 'Anti-inflammatory', 'Digestive Health', 'Immune Support'],
    'Nelavembu': ['Detoxification', 'Digestive Health', 'Immune Support', 'Fever Relief', 'Anti-inflammatory'],
    'Nerale': ['Digestive Health', 'Anti-inflammatory', 'Detoxification', 'Respiratory Conditions', 'Joint Pain'],
    'Nooni': ['Detoxification', 'Digestive Health', 'Heart Health', 'Liver Health', 'Anti-inflammatory'],
    'Onion': ['Digestive Health', 'Heart Health', 'Immune Support', 'Skin Conditions', 'Anti-inflammatory'],
    'Padri': ['Digestive Health', 'Skin Conditions', 'Anti-inflammatory', 'Detoxification', 'Respiratory Conditions'],
    'Palak(Spinach)': ['Bone Health', 'Digestive Health', 'Eye Health', 'Anti-inflammatory', 'Heart Health'],
    'Papaya': ['Digestive Health', 'Immune Support', 'Anti-inflammatory', 'Heart Health', 'Skin Conditions'],
    'Parijatha': ['Detoxification', 'Digestive Health', 'Anti-inflammatory', 'Skin Conditions', 'Respiratory Conditions'],
    'Pea': ['Digestive Health', 'Heart Health', 'Immune Support', 'Bone Health', 'Anti-inflammatory'],
    'Pepper': ['Anti-inflammatory', 'Immune Support', 'Respiratory Health', 'Digestive Health', 'Metabolism Boost'],
    'Pomoegranate': ['Digestive Health', 'Heart Health', 'Skin Conditions', 'Immune Support', 'Anti-inflammatory'],
    'Pumpkin': ['Digestive Health', 'Heart Health', 'Eye Health', 'Anti-inflammatory', 'Immune Support'],
    'Raddish': ['Joint Pain', 'Digestive Health', 'Anti-inflammatory', 'Detoxification', 'Respiratory Conditions'],
    'Rose': ['Wound Healing', 'Heart Health', 'Skin Conditions', 'Anti-inflammatory', 'Digestive Health'],
    'Sampige': ['Digestive Health', 'Joint Pain', 'Anti-inflammatory', 'Skin Conditions', 'Respiratory Conditions'],
    'Sapota': ['Digestive Health', 'Heart Health', 'Skin Conditions', 'Anti-inflammatory', 'Immune Support'],
    'Seethaashoka': ['Detoxification', 'Digestive Health', 'Anti-inflammatory', 'Women\'s Health', 'Skin Conditions'],
    'Seethapala': ['Anti-inflammatory', 'Heart Health', 'Immune Support', 'Digestive Health', 'Skin Conditions'],
    'Spinach1': ['Eye Health', 'Digestive Health', 'Bone Health', 'Anti-inflammatory', 'Heart Health'],
    'Tamarind': ['Digestive Health', 'Anti-inflammatory', 'Heart Health', 'Liver Health', 'Immune Support'],
    'Taro': ['Anti-inflammatory', 'Joint Pain', 'Digestive Health', 'Heart Health', 'Skin Conditions'],
    'Tecoma': ['Anti-inflammatory', 'Digestive Health', 'Respiratory Conditions', 'Detoxification', 'Joint Pain'],
    'Thumbe': ['Respiratory Conditions', 'Digestive Health', 'Anti-inflammatory', 'Joint Pain', 'Skin Conditions'],
    'Tomato': ['Digestive Health', 'Heart Health', 'Eye Health', 'Anti-inflammatory', 'Skin Conditions'],
    'Tulsi': ['Anti-inflammatory', 'Respiratory Health', 'Digestive Health', 'Stress Relief', 'Immune Support'],
    'Turmeric': ['Digestive Health', 'Skin Conditions', 'Anti-inflammatory', 'Joint Pain', 'Immune Support'],
    'ashoka': ['Detoxification', 'Digestive Health', 'Anti-inflammatory', 'Women\'s Health', 'Skin Conditions'],
    'camphor': ['Cough Relief', 'Stress Relief', 'Respiratory Health', 'Anti-inflammatory', 'Joint Pain'],
    'kamakasturi': ['Respiratory Conditions', 'Anti-inflammatory', 'Digestive Health', 'Stress Relief', 'Skin Conditions'],
    'kepala': ['Digestive Health', 'Heart Health', 'Anti-inflammatory', 'Joint Pain', 'Respiratory Conditions']
    }
    
    
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
    cure = plant_cure[predicted_class]

    return predicted_class, confidence , cure


@app.post("/predict")
async def predict_image(image: UploadFile):
    # Create temporary directory for uploaded image
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded image to temporary file
        img_path = f"{temp_dir}/temp.jpg"
        with open(img_path, "wb") as f:
            f.write(await image.read())

        # Predict class and confidence for the saved image
        predicted_class, confidence, cure = predict_single_image(model, img_path)

        # Delete temporary image and directory
        os.remove(img_path)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "plant_cure": cure
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=8000)
