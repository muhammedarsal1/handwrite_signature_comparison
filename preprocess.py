import os
import cv2
import numpy as np
import random

# Set absolute dataset paths
PROJECT_ROOT = r"C:\Users\anshi\OneDrive\Desktop\my_project"
SIGNATURE_PATH = os.path.join(PROJECT_ROOT, "data", "Signature")
FORG_PATH = os.path.join(SIGNATURE_PATH, "full_forg")
ORG_PATH = os.path.join(SIGNATURE_PATH, "full_org")
HANDWRITING_PATH = os.path.join(PROJECT_ROOT, "data", "HandWrite")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data")

# Target image size
IMAGE_SIZE = (128, 128)

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Function to load and preprocess a single image
def preprocess_image(image_path, target_size=IMAGE_SIZE):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # Shape: (128, 128, 1)
    return img

# Function to create signature pairs for Siamese training
def create_signature_pairs(org_path, forg_path):
    org_files = [f for f in os.listdir(org_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    forg_files = [f for f in os.listdir(forg_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    pairs = []
    labels = []

    # Positive pairs (Original vs Original)
    for _ in range(100):
        org1, org2 = random.sample(org_files, 2)
        img1 = preprocess_image(os.path.join(org_path, org1))
        img2 = preprocess_image(os.path.join(org_path, org2))
        pairs.append([img1, img2])
        labels.append(1)

    # Negative pairs (Original vs Forged)
    for _ in range(100):
        org = random.choice(org_files)
        forg = random.choice(forg_files)
        img1 = preprocess_image(os.path.join(org_path, org))
        img2 = preprocess_image(os.path.join(forg_path, forg))
        pairs.append([img1, img2])
        labels.append(0)

    pairs = np.array(pairs)
    labels = np.array(labels)
    return pairs, labels

# Function to preprocess handwriting images and assign labels
def preprocess_handwriting_data(handwriting_path):
    images = []
    labels = []
    
    folders = sorted(os.listdir(handwriting_path))
    label_dict = {folder: idx for idx, folder in enumerate(folders)}

    for folder in folders:
        folder_path = os.path.join(handwriting_path, folder)
        if os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for file in files:
                img_path = os.path.join(folder_path, file)
                img = preprocess_image(img_path)
                images.append(img)
                labels.append(label_dict[folder])

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Process and save data
signature_pairs, signature_labels = create_signature_pairs(ORG_PATH, FORG_PATH)
handwriting_images, handwriting_labels = preprocess_handwriting_data(HANDWRITING_PATH)

np.save(os.path.join(OUTPUT_PATH, "sig_pairs.npy"), signature_pairs)
np.save(os.path.join(OUTPUT_PATH, "sig_labels.npy"), signature_labels)
np.save(os.path.join(OUTPUT_PATH, "hw_data.npy"), handwriting_images)
np.save(os.path.join(OUTPUT_PATH, "hw_labels.npy"), handwriting_labels)

print(f"Signature pairs saved: {signature_pairs.shape}")
print(f"Handwriting data saved: {handwriting_images.shape}")