# Diabetic Retinopathy Detection using DenseNet-121

## Overview
This project is part of the Infyma AI Hackathon 2025, where we developed a deep learning model to detect diabetic retinopathy using DenseNet-121. The model is trained on a balanced dataset and fine-tuned for better accuracy.

## Dataset
The dataset consists of fundus images of different severity levels of diabetic retinopathy. The images are divided into:
- **Train**: Used for model training.
- **Test**: Used for model evaluation.

## Model Architecture
We used **DenseNet-121**, a pre-trained deep learning model, as the backbone for feature extraction. The final layers were fine-tuned for classification.

## Training Strategy
1. **Initial Training**: The DenseNet-121 base model was frozen, and only the new classifier layers were trained.
2. **Fine-Tuning**: Some layers of the DenseNet-121 model were unfrozen, and the model was trained further with a lower learning rate.

## Data Augmentation
To improve generalization, the following augmentations were applied:
- Rotation
- Width & height shifts
- Shear & zoom transformations
- Horizontal flipping

## Training Process
- **Epochs**: 30
- **Batch Size**: 64 for training, 32 for testing
- **Optimizer**: Adam with a learning rate of 0.0001 (reduced for fine-tuning)
- **Loss Function**: Categorical Crossentropy

## Evaluation
The model was evaluated using:
- Accuracy
- Loss on test data

## How to Run
1. Clone the repository:
   ```sh
   git clone https://github.com/mina-hill/infyma_ai_hackathon.git
   ```
2. Navigate to the directory:
   ```sh
   cd infyma_ai_hackathon
   ```
3. Run the training script (if using Google Colab, first upload the dataset):
   ```sh
   python train.py
   ```

## Testing the Model
To test the model, randomly select an image from the test dataset and run:
```sh
python test_model.py --image_path <path_to_image>
```
### Team Name
- **BYTE BRAINED**

## Authors
- **Minahil Kashif** **minahilkashif10@gmail.com**
- **Areesha Hussain** **i230123@isb.nu.edu.pk**
- Infyma AI Hackathon 2025 Team

## Acknowledgments
- Kaggle for providing the dataset.
- TensorFlow & Keras for deep learning frameworks.


 
