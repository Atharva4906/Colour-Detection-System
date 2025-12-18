import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd

app = Flask(__name__)

# Set up the upload folder
UPLOAD_FOLDER = r"D:\2nd_year\1st_sem\kaargar\K\Colour-Detection-System\uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define a route to serve the uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Function to get color name from CSV (color detection)
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv(r'C:\Users\Admin\Downloads\PYTHON\colors.csv', names=index, header=None)

def get_color_name(R, G, B):
    minimum = 1000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in the request'
    file = request.files['file']
    if file.filename == '':
        return 'No file selected for uploading'
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_image', filename=filename))

# Route to handle the uploaded image and display color when clicked
@app.route('/uploaded_image/<filename>', methods=['GET', 'POST'])
def uploaded_image(filename):
    if request.method == 'POST':
        # Get x and y coordinates from the form
        x = int(request.form['x'])
        y = int(request.form['y'])

        # Load the uploaded image
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = cv2.imread(img_path)

        # Get the RGB values at the clicked coordinates
        b, g, r = img[y, x]

        # Ensure the values for b, g, r are integers
        b, g, r = map(int, [b, g, r])

        # Check that b, g, r are within the valid range (0 to 255)
        b = max(0, min(255, b))
        g = max(0, min(255, g))
        r = max(0, min(255, r))

        # Get the color name
        color_name = get_color_name(r, g, b)

        # Add the RGB and color name on the image
        cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)
        text = f"{color_name} R={r} G={g} B={b}"
        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Save the modified image
        modified_filename = 'modified_' + filename
        modified_img_path = os.path.join(app.config['UPLOAD_FOLDER'], modified_filename)
        cv2.imwrite(modified_img_path, img)

        # Redirect to show the modified image
        return redirect(url_for('uploaded_image', filename=modified_filename))

    return render_template('uploaded_image.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)

# import os
# #import requests
# from flask import Flask, render_template, request, redirect, url_for, send_from_directory
# from collections import Counter
# #from sklearn.cluster import KMeans
# import numpy as np
# #from PIL import Image

# app = Flask(__name__)

# # Configure the upload folder
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Ensure the uploads folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def extract_dominant_color(image_path, k=1):
#     """Extract the dominant color from an image."""
#     image = Image.open(image_path)
#     image = image.resize((image.width // 10, image.height // 10))  # Reduce image size for speed
#     pixels = np.array(image).reshape(-1, 3)
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(pixels)
#     dominant_color = kmeans.cluster_centers_[0].astype(int)
#     return tuple(dominant_color)

# def compare_rgb(rgb1, rgb2):
#     """Compare RGB values and calculate the Euclidean distance."""
#     diff = sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5
#     return diff

# @app.route('/compare/<filename>', methods=['GET', 'POST'])
# def compare_images(filename):
#     uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#     if request.method == 'POST':
#         # Get the Amazon image URL from the form
#         amazon_image_url = request.form.get('amazon_image_url')
        
#         # Fetch the Amazon image
#         response = requests.get(amazon_image_url, stream=True)
#         amazon_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'amazon_image.jpg')
#         if response.status_code == 200:
#             with open(amazon_image_path, 'wb') as f:
#                 f.write(response.content)
#         else:
#             return "Failed to fetch the Amazon image."

#         # Analyze and compare colors
#         uploaded_colors = extract_dominant_color(uploaded_image_path)
#         amazon_colors = extract_dominant_color(amazon_image_path)

#         # Convert the RGB colors to a format that can be easily compared
#         uploaded_rgb = tuple(map(int, uploaded_colors))
#         amazon_rgb = tuple(map(int, amazon_colors))

#         # Compare the RGB values (simple comparison based on difference)
#         color_difference = compare_rgb(uploaded_rgb, amazon_rgb)

#         return render_template(
#             'compare.html',
#             uploaded_image=filename,
#             amazon_image='amazon_image.jpg',
#             uploaded_colors=uploaded_colors,
#             amazon_colors=amazon_colors,
#             color_difference=color_difference
#         )

#     # If it's a GET request, show the form for Amazon image URL
#     return render_template('amazon_input.html', uploaded_image=filename)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return 'No file part in the request'
#     file = request.files['file']
#     if file.filename == '':
#         return 'No file selected for uploading'
#     if file:
#         filename = file.filename
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         return redirect(url_for('compare_images', filename=filename))

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)

# import os
# import requests
# from flask import Flask, render_template, request, redirect, url_for, send_from_directory
# from collections import Counter
# from sklearn.cluster import KMeans
# import numpy as np
# from PIL import Image

# app = Flask(__name__)

# # Configure the upload folder
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Ensure the uploads folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def extract_dominant_color(image_path, k=1):
#     """Extract the dominant color from an image."""
#     image = Image.open(image_path)
#     image = image.resize((image.width // 10, image.height // 10))  # Reduce image size for speed
#     pixels = np.array(image).reshape(-1, 3)
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(pixels)
#     dominant_color = kmeans.cluster_centers_[0].astype(int)
#     return tuple(dominant_color)

# def compare_rgb(rgb1, rgb2):
#     """Compare RGB values and calculate the Euclidean distance."""
#     diff = sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5
#     return diff

# @app.route('/')
# def home():
#     return render_template('index.html')  # Renders the HTML form for image upload

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return 'No file part in the request'
#     file = request.files['file']
#     if file.filename == '':
#         return 'No file selected for uploading'
#     if file:
#         filename = file.filename
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         return redirect(url_for('amazon_input', filename=filename))

# @app.route('/amazon_input/<filename>', methods=['GET', 'POST'])
# def amazon_input(filename):
#     """Render the page to input the Amazon image URL."""
#     if request.method == 'POST':
#         # Get the Amazon image URL from the form
#         amazon_image_url = request.form.get('amazon_image_url')
        
#         # Fetch the Amazon image
#         response = requests.get(amazon_image_url, stream=True)
#         amazon_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'amazon_image.jpg')
#         if response.status_code == 200:
#             with open(amazon_image_path, 'wb') as f:
#                 f.write(response.content)
#         else:
#             return "Failed to fetch the Amazon image."

#         # Analyze and compare colors
#         uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         uploaded_colors = extract_dominant_color(uploaded_image_path)
#         amazon_colors = extract_dominant_color(amazon_image_path)

#         # Convert the RGB colors to a format that can be easily compared
#         uploaded_rgb = tuple(map(int, uploaded_colors))
#         amazon_rgb = tuple(map(int, amazon_colors))

#         # Compare the RGB values (simple comparison based on difference)
#         color_difference = compare_rgb(uploaded_rgb, amazon_rgb)

#         return render_template(
#             'compare.html',
#             uploaded_image=filename,
#             amazon_image='amazon_image.jpg',
#             uploaded_colors=uploaded_colors,
#             amazon_colors=amazon_colors,
#             color_difference=color_difference
#         )


# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)

