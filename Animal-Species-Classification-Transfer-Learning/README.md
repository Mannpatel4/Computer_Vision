# Animal Species Classification (Transfer Learning)

##  Project Overview
This project classifies various animal species using advanced Computer Vision techniques. To achieve high accuracy and fast training times on a limited dataset, I utilized **Transfer Learning** with the **MobileNetV2** architecture, transitioning from a basic CNN approach to an industry-standard methodology.

##  Tech Stack
* **Language:** Python
* **Framework:** TensorFlow / Keras
* **Pre-trained Model:** MobileNetV2 (ImageNet weights)

## Approach
To optimize this model, I applied a rigorous fine-tuning strategy:
1. **Base Model:** Loaded MobileNetV2 and discarded the top classification layer.
2. **Feature Extraction:** Froze the base layers to retain pre-learned features (basic shapes and textures from ImageNet).
3. **Selective Unfreezing:** Unfroze the top 30 layers and trained the model using a highly controlled, low learning rate (`Adam(1e-5)`). This adapted the deeper layers specifically to animal features without destroying the pre-trained weights.
4. **Custom Head:** Added a `GlobalAveragePooling2D` layer and a final `Dense` softmax layer for species classification.

##  Results
* Successfully prevented overfitting using `EarlyStopping` and Data Augmentation.
* The fine-tuned MobileNetV2 model significantly outperformed standard CNN architectures, proving the efficiency of transfer learning for complex image classification tasks.