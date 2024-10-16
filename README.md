
# Nationwide Farmland Identification Dataset - CNN

This repository contains essential Python scripts for the **Nationwide Farmland Identification Dataset** project, which utilizes geospatial data and machine learning to classify farmland and non-farmland areas using satellite imagery. This project leverages both a **Convolutional Neural Network (CNN)** and META's **Segment Anything Model (SAM)** for comprehensive land analysis.

## Project Scripts

### 1. **Moving_dataset.py**
   - **Purpose:** This script is responsible for preparing and organizing the dataset for model training and validation. It ensures that the image data is properly categorized into farmland (positive) and metropolitan (negative) examples, as well as splitting the data into training and validation sets.
   - **Main Functionality:**
     - Automates the movement of images between directories to create structured datasets required for both CNN and SAM models.
   - **Key Functions:**
     - `move_files(src, dest)`: Moves images from the source directory to the correct destination for structured dataset preparation.
   - **Usage Example:**
     ```bash
     python Moving_dataset.py --src /path/to/dataset --dest /path/to/output
     ```

### 2. **Testing_CNN.py**
   - **Purpose:** This script handles the testing and evaluation of the CNN model on the validation dataset. After training the classifier, this script will assess its accuracy and performance in classifying farmland and metropolitan areas.
   - **Main Functionality:**
     - Loads a trained CNN model and performs evaluation using the validation dataset.
     - Computes key performance metrics such as accuracy, precision, recall, and confusion matrices.
   - **Key Functions:**
     - `evaluate_model(model, test_data)`: Evaluates the trained CNN on the test set.
     - `plot_confusion_matrix(cm, classes)`: Displays a confusion matrix for detailed performance analysis.
   - **Usage Example:**
     ```bash
     python Testing_CNN.py --model /path/to/model --test_data /path/to/test/data
     ```

### 3. **cnn_model.py**
   - **Purpose:** This script defines the architecture and training process of the CNN model. It builds a convolutional neural network for image classification, specifically designed to classify farmland and metropolitan areas based on satellite imagery.
   - **Main Functionality:**
     - Constructs the CNN model, prepares the data pipeline, and trains the classifier.
     - Integrates data augmentation techniques to improve model generalization.
     - Saves the trained model for future testing or deployment.
   - **Key Functions:**
     - `build_model()`: Defines the structure of the CNN, including convolutional layers, activation functions, and pooling.
     - `train_model(model, train_data, val_data)`: Trains the CNN on the specified training data, using validation data to monitor performance.
     - `save_model(model, path)`: Saves the trained CNN to the specified path for later use.
   - **Usage Example:**
     ```bash
     python cnn_model.py --train_data /path/to/train --val_data /path/to/val --output_model /path/to/output
     ```

## Parallel Integration of CNN and SAM

This project leverages **parallel processing** by combining two powerful models:
- **CNN Model:** The CNN is responsible for classifying satellite images into two categories: farmland (positive) and metropolitan (negative). The CNN uses a set of training data to learn how to distinguish between these two types of land.
- **Segment Anything Model (SAM):** META's SAM model is used in parallel with the CNN to provide detailed image segmentation. This model identifies key regions in satellite images that are likely to represent farmland or other features, enhancing the CNN’s performance.

### Workflow:
1. **Dataset Preparation:**
   - Images are first organized and categorized using the `Moving_dataset.py` script to ensure the dataset is correctly structured for both the CNN and SAM models.
2. **Segmentation with SAM:**
   - The SAM model is run on the prepared satellite imagery to generate segmentation masks. These masks help the CNN by highlighting important regions within the images.
3. **CNN Classification:**
   - The CNN model is then trained using the segmented images to classify each image as either farmland or metropolitan area. This combination of segmentation and classification improves the overall model accuracy.
4. **Testing and Validation:**
   - The `Testing_CNN.py` script evaluates the performance of the trained model, providing insights into its classification accuracy and areas for improvement.

---

