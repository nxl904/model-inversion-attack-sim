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







