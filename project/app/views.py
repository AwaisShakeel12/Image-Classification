from django.shortcuts import render
import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load your pre-trained model
model = load_model(r'E:\django\django_x_deep_learning\images_classifier\project\app\Model\best.keras')

# Function to preprocess images
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)  # Resize image
    img_array = img_to_array(img) / 255.0  # Normalize image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# View for the home page
def home(request):
    if request.method == 'POST':
        input_folder = request.POST.get('input_folder')
        output_folder = request.POST.get('output_folder')

        # Validate input folder
        if not os.path.isdir(input_folder):
            return render(request, 'home2.html', {'error': f"Input folder '{input_folder}' does not exist."})
        
        # Validate or create output folder
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                return render(request, 'home2.html', {'error': f"Unable to create output folder: {str(e)}"})

        # Process images
        image_count = 0
        try:
            for image_name in os.listdir(input_folder):
                image_path = os.path.join(input_folder, image_name)
                if not image_name.lower().endswith(('png', 'jpg', 'jpeg')):  # Skip non-image files
                    continue

                # Preprocess and predict
                image = preprocess_image(image_path, target_size=(128, 128))
                prediction = model.predict(image)
                predicted_class = np.argmax(prediction)  # Get the index of the predicted class

                # Create a folder for the predicted class if it doesn't exist
                class_folder = os.path.join(output_folder, f'class_{predicted_class}')
                os.makedirs(class_folder, exist_ok=True)

                # Move the image to the corresponding folder
                shutil.copy(image_path, os.path.join(class_folder, image_name))
                image_count += 1

        except Exception as e:
            return render(request, 'home2.html', {'error': f"An error occurred during processing: {str(e)}"})

        return render(request, 'home2.html', {'success': f"{image_count} images classified successfully!"})

    return render(request, 'home2.html')
