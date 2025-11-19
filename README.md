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

# Technologies used 

We will be using the following python libraries and associated technologies to accomplish this: 

**tensorflow/keras:** deep-learning framework used to compile and build the DNN. tensorflow is essentially a machine-learning engine 

**numpy:** python's numerical computing library, used to create synthetic MFCC feature vectors. It's being used in this context to generate and manipulate the simulated dataset

**scikit-learn:** python's machine-learning utility toolkit

**matplotlib:** python's plotting library. It will be used to split data into training/testing sets

# Methods used

**MFCC (Mel-Frequency Cepstral Coefficients):** audio feature used in speech/speaker recognition

**ReLu:** Math fucition used inside the model to help it learn patterns

**One-Hot Encoding:** Categorical Labels, coverts numeric class ID's into one-hot vectors. Used for training classification models

**Deep Neural Network (DNN):**  Takes MFCC vectors as inputs, and predicts which speaker the voice belongs to. 

**Softmax Classification:** Turns raw scores into probabilities acorss 10 voice samples (MFCC vectors) 

**Adam Optimizer:** Updates the DNN's weights while training

**Categorical Cross-Entropy Loss:** Measures how far the model's predictions are from the true speaker labels


# Phase 1: Building a vulnerable AI 


For the first phase we will prepare the victim system for the attack. Our objective in this phase is to create a confident AI model that has memorized sensitive data. The code snippet below breaks down Phase 1 into 2 distinct steps needed
to build the vulnerable (target) AI model. 

1. Select simulated MFCC Dataset
2. Train the target AI model

Since this is a simulation for academic purposes, we will create random data to mimic Mel-Frequency Cepstral Coefficient (MFCC) features. MFCC is a feature extraction process that turns complex sound waves into numeric values. Essentially, MFCC is 
being used in this context to be the numerical representation of a voice. Once we have the MFCC dataset parameters we will train a Deep Neural Network (DNN - a type of broader AI ) that can identify the voices (MFCC values) that we generated in step 1. 

# **Technologies used in Phase 1**: 

**TensorFlow/Keras:** trains and builds the neural network

**NumPy:** creates fake MFCC data

# **Methods used (what's being done) in Phase 1** : 

**MFCC simulation:** pretends to have human voice features

**Label encoding:** turns the users IDs into 0/1 vectors

**Neuarl network creation:** building a simple DNN that can tell users aparat

**Model training:** teaching the victim model which MFCCs belong to which user 

**Selection of target user:** picks user 5 (random) as a the person that we will "attack"


```python

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

```








