# Model Inversion Attack Simulation

In this simlulation we will be replicating simple Model Inversion Attack (MIA). The simulated attack will be used against a basic Convolutional Nueral Network (CNN) that we will train using synthetically generated Mel-Frequency Cepstral Coefficients (MFCC) to simulate a users voice. This CNN will represent an institutions AI engine specfically trained to recognize a users voice for phone authentication in an Intellignet Voice Recognition (IVR) phone tree.

This simulation is being done for academic purposes using sythentically generated voice signatures to better understand how MIA's work and how to implement basic defenses to mitigate them. 

# Scenario
Imagine a bank's IVR system uses your unique voiceprint (like a fingerprint, but for your voice) to authenticate you.

The Model Inversion Attack doesn't steal the bank's database; it targets the AI brain that was trained to recognize your voice. This AI brain, after being trained, inadvertently keeps a high-resolution "picture" of your voiceprint in its memory.

The attack works by showing the AI random noise and telling it, "Make this noise sound like User ID 999." Using mathematical feedback, the noise is slowly sculpted, pixel-by-pixel, until it perfectly matches the unique vocal characteristics of User ID 999.

The result is a cloned voice feature vector that can be turned into a playable audio file to impersonate the victim and bypass security. 

# Approach

This simulation consists of 4 distinct phases and associated code buckets for reference. The phases are as follows: 

1. Building a vulnerable AI : trains a voice-recognition model
2. Preparing the attack tool: sets up an attack by creating a random vector 
3. Running the attack: tweaks that random vector (phase 2) until the model thinks it's User 5
4. Evaluation and analysis: checks if the attack worked and compares orginal vs reconstructed features

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

**Technologies used in Phase 1**: 

*TensorFlow/Keras:* trains and builds the neural network

*NumPy:* creates fake MFCC data

**Methods used (what's being done) in Phase 1** : 

*MFCC simulation:* pretends to have human voice features

*Label encoding:* turns the users IDs into 0/1 vectors

*Neuarl network creation:* building a simple DNN that can tell users aparat

*Model training:* teaching the victim model which MFCCs belong to which user 

*Selection of target user:* picks user 5 (random) as a the person that we will "attack"


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

# Output of Phase 1:

```python
#Phase I: System Setup - Simulating MFCC Data...
#Data Simulated. Target (User ID 5) Feature Vector shape: (390,)
#Phase I: Starting Target DNN Model Training...
#Phase I: DNN Model Training Complete.
```

# Phase 2: Building the Attack Tool

The goal of phase 2 is to establish the exact targete (victim model) and begin optimization process. Optimization initialization we're looking at how we're going to setup the "problem" for the computer to solve. Think of phase 2 as
setting the coordinates on a GPS and then checking that you have fuel and a starting position for your car before the trip. The difference is that we're driving to a specfic destination that we don't know yet. The "where" this destination is 
will be part of phase 3 (running the attack). 

**Technologies used in Phase 2:**

*TensorFlow:* creates a variable for the random starting vector

*Adam Optimizer:* corrects the fake MFCC during the simulated attack 

*NumPy:* reshapes data and prepares the inputs

**Methods used in Phase 2**

*Create the target label:* We want the model to output **User 5**

*Random initialization:* starts with noise, not real audio

*Define the hyperparameters:* the number of interations, learning rate, etc...

*Set up optimization:* preparing to tweak the vector to look like the output (MFCC syntentic voice) of User 5

```python
#phase 2: start simulated attack (optimization initialization) 

# --- Step 3: Define Target Output ---
# Target vector for User ID 5
y_target = tf.keras.utils.to_categorical(target_class_id, num_classes=NUM_CLASSES)
y_target = np.expand_dims(y_target, axis=0) 

# --- Step 4: Initialize Random Input (Corrected) ---
input_shape = (1, VECTOR_LENGTH) 
# Create a random noise vector for the MFCC features (Standard Normal Distribution)
x_rand = tf.Variable(
    initial_value=tf.random.normal(shape=input_shape, mean=0.0, stddev=1.0, dtype=tf.float32), 
    dtype=tf.float32
)

# --- Step 5: Define Hyperparameters ---
LEARNING_RATE = 0.1       # Higher LR might be needed for vector optimization
REGULARIZATION_WEIGHT = 0.005 # Lambda (λ) for the prior loss
MAX_ITERATIONS = 5000     

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

print("Phase II: Optimization Setup Complete. Ready to execute MFCC inversion attack.")
```

# Output of Phase 2
```python
#Phase II: Optimization Setup Complete. Ready to execute MFCC inversion attack.
```

# Phase 3 - Running the Attack 

**Technologies used in Phase 3**

*TensorFlow GradientTape:* computes how to change vector
*TensorFlow optimiziers:* applies the gradient updates
*TensforFlow loss functions:* calculates proximity to target

**Methods used in Phase 3**

*Confidence loss:* pushes fake MFCC to make the model say: "This is User 5."
*Regularization (L2 penalty):* prevents the fake vector from becoming unrealistic
*Loss function combination:* Mixes "look like User 5" with "stay realistic
*Gradient descent on the input:* instead of training the model, you "train the input vector" to fool it. 
*Iterative optimization:* Slowly updade the fake MFCC thousands of times until it mimics User 5. 
*Reconstruction output:* Final optimized MFCC vector 

```python
#Phase 3: Core Attack (Iterative Reconstruction)

# --- Step 6: Define Loss Function (L) ---
def loss_fn(model, x, y_target, regularization_weight):
    # Confidence Loss: Forces the input towards the target class
    confidence_loss = tf.keras.losses.categorical_crossentropy(y_target, model(x))
    
    # Prior Loss (Regularization): Keeps the feature vector values constrained
    prior_loss = tf.norm(x) # L2 norm on the input vector
    
    total_loss = confidence_loss + (regularization_weight * prior_loss)
    return total_loss

# --- Steps 7, 8, & 9: Optimization Loop ---
print("Phase III: Starting Iterative Reconstruction (MFCC Optimization)...")

for k in range(MAX_ITERATIONS):
    # Step 7: Calculate Gradient (using GradientTape)
    with tf.GradientTape() as tape:
        tape.watch(x_rand)
        L = loss_fn(model, x_rand, y_target, REGULARIZATION_WEIGHT)
    
    gradient = tape.gradient(L, x_rand)
    
    # Step 8: Update Input (Optimization Step)
    optimizer.apply_gradients([(gradient, x_rand)])
    
    # Optional Progress Print
    if (k + 1) % 1000 == 0:
        print(f"Iteration {k+1}/{MAX_ITERATIONS}, Loss: {L.numpy().mean():.4f}")

x_reconstructed_vector = x_rand.numpy().squeeze()
print("Phase III: MFCC Feature Vector Reconstruction Complete.")

```

# Phase 3 Output

```python

'''Phase III: Starting Iterative Reconstruction (MFCC Optimization)...
Iteration 1000/5000, Loss: 0.0254
Iteration 2000/5000, Loss: 0.0254
Iteration 3000/5000, Loss: 0.0254
Iteration 4000/5000, Loss: 0.0254
Iteration 5000/5000, Loss: 0.0254
Phase III: MFCC Feature Vector Reconstruction Complete.'''

```

# Phase 4 Evaluation & Analysis

This phase confirm the attack was succussful. We will seek to verify the reconstructued features are accurate and take the output MFCC and translate it to a voice. The intention being the end result is a playable voice that could be 
used in a nafarious authorization. 


**Technologies used in Phase 4**

*Matplotlib:* draws the comparison chart

*NumPy:* reshapes the reconstructed vector for prediction 

*TensorFlow:* runs the prediction 

**Methods used in Phase 4**

*Plotting orginal vs reconstructed MFCC:* This is a visual check to see if they look similiar

*Confidence check:* We ask the victim model "Do you think this new vector is User 5?"

*Attack success test:* If the model gives >90% confidence AND selects User 5 = success

```python
#Phase 4: Evaluation & analysis

# --- Step 10: Visualize Fidelity (Conceptual) ---
# Since this is a high-dimensional vector, we plot the values.
plt.figure(figsize=(10, 4))
plt.plot(x_true, label='Original MFCC Vector (x_true)', alpha=0.7)
plt.plot(x_reconstructed_vector, label='Reconstructed MFCC Vector (x*)', linestyle='--', alpha=0.7)
plt.title(f"Comparison of Original vs. Reconstructed MFCC Vectors for User ID {target_class_id}")
plt.xlabel("Feature Index")
plt.ylabel("MFCC Value (Amplitude)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- Step 11: Verify Model Confidence ---
# Check what the target model (F) predicts for the reconstructed vector
x_check = np.expand_dims(x_reconstructed_vector, axis=0)
predictions = model.predict(x_check, verbose=0)
confidence = predictions[0, target_class_id] * 100
max_conf_class = np.argmax(predictions)

print(f"\nPhase IV: Model Confidence Check")
print(f"Target Class ID: {target_class_id}")
print(f"Model's highest prediction: Class {max_conf_class}")
print(f"Model Confidence in Target Class for x*: {confidence:.2f}%")

# --- Step 12: Synthesis Step (Conceptual) ---
if confidence > 90 and max_conf_class == target_class_id:
    print("\nAttack Successful! The reconstructed vector is highly confident for the target user.")
    print("Next step is to feed the x* vector into a VOCDER for voice synthesis.")
else:
    print("\nAttack failed or needs more iteration/tuning.")
```
# Output of Phase 4 

<img width="1216" height="551" alt="image" src="https://github.com/user-attachments/assets/1112f847-de21-4321-91de-c8ac02e0c31d" />


```python
'''
Phase IV: Model Confidence Check
Target Class ID: 5
Model's highest prediction: Class 5
Model Confidence in Target Class for x*: 99.70%

Attack Successful! The reconstructed vector is highly confident for the target user.
Next step is to feed the x* vector into a VOCDER for voice synthesis.
'''
```









