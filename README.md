# Model Inversion Attack Simulation

In this simlulation we will be replicating simple Model Inversion Attack (MIA). The simulated attack will be against a basic Convolutional Nueral Network (CNN) that we will train using publically available NIST data. This CNN will represent 
a institutions Intelligent Voice Recognition (IVR) phone tree. 

This simulation is being done for academic purposes using publically available data to better understand how MIA's work and how to implement basic defenses to mitigate them. 

# Scenario
Imagine a bank's IVR system uses your unique voiceprint (like a fingerprint, but for your voice) to authenticate you.

The Model Inversion Attack doesn't steal the bank's database; it targets the AI brain that was trained to recognize your voice. This AI brain, after being trained, inadvertently keeps a high-resolution "picture" of your voiceprint in its memory.

The attack works by showing the AI random noise and telling it, "Make this noise sound like User ID 999." Using mathematical feedback, the noise is slowly sculpted, pixel-by-pixel, until it perfectly matches the unique vocal characteristics of User ID 999.

The result is a cloned voice feature vector that can be turned into a playable audio file to impersonate the victim and bypass security. 

# Approach

This simulation consists of 4 distinct phases and associated code buckets for reference. The phases are as follows: 

1. Building a vulnerable AI
2. Preparing the attack tool
3. Running the attack
4. Evaluation and analysis

# Phase 1: Building a vulnerable AI 


For the first phase we will prepare the victim system for the attack. Our objective in this phase is to create a confident AI model that has memorized sensitive data. 

''''
'''

# phase 1: setup up target voice model 

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define MFCC/Voice Parameters (Simplified for simulation)
NUM_MFCCS = 39 # A common number (13 static + 13 delta + 13 delta-delta)
NUM_CLASSES = 10 # Assume 10 unique speakers/user IDs
VECTOR_LENGTH = NUM_MFCCS * 10 # Assume 10 frames are concatenated for an utterance vector

# --- Step 1: Select Simulated MFCC Dataset ---
print("Phase I: System Setup - Simulating MFCC Data...")

# SIMULATION: Create synthetic MFCC data for 10 users (classes)
X_simulated = np.random.randn(500, VECTOR_LENGTH).astype('float32')
y_simulated = np.repeat(np.arange(NUM_CLASSES), 500 // NUM_CLASSES)
y_simulated = tf.keras.utils.to_categorical(y_simulated, num_classes=NUM_CLASSES)

# Define Target (User ID 5) and isolate the ground truth sample
target_class_id = 5
idx_true = np.where(np.argmax(y_simulated, axis=1) == target_class_id)[0][0]
x_true = X_simulated[idx_true] # The 'private' MFCC vector we aim to reconstruct

# Reshape the ground truth for input (add batch dimension)
x_true_input = np.expand_dims(x_true, axis=0) 
print(f"Data Simulated. Target (User ID {target_class_id}) Feature Vector shape: {x_true.shape}")



# --- Step 2: Train the Target Model (F) ---
def create_dnn_model(input_dim, num_classes):
    # A simple Deep Neural Network (DNN) for speaker identification
    model = models.Sequential([
        tf.keras.Input(shape=(input_dim,)), 
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = create_dnn_model(VECTOR_LENGTH, NUM_CLASSES)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Phase I: Starting Target DNN Model Training...")
# Train the model (F) - this is the "vulnerable" step
model.fit(
    X_simulated, y_simulated,
    epochs=10,
    batch_size=32,
    verbose=0 
)
print("Phase I: DNN Model Training Complete.")

'''
''''







