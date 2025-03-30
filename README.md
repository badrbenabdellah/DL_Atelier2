<h1 align="left">Lab 2: Computer Vision with PyTorch</h1>

###

<p align="left">This project is a Jupyter notebook implementing a convolutional neural network (CNN) for MNIST image classification using PyTorch.</p>

###

<h2 align="left">Features</h2>

###

<p align="left">🖼️ MNIST Classification: Tackles digit recognition with 5 models (CNN, Faster R-CNN, VGG16, AlexNet, ViT).<br>📊 Metrics Galore: Accuracy, F1 score, loss, and training time for all models.<br>⚡ GPU Power: Leverages Colab’s T4 GPU for speedy training.<br>🔧 Fine-Tuning: Pretrained VGG16 and AlexNet adjusted for MNIST (1-channel, 10 classes).<br>🧠 From Scratch: ViT built from the ground up—patches, transformers, and all!</p>

###

<h2 align="left">Prérequis</h2>

###

<p align="left">Make sure you have installed the necessary libraries before running the notebook:<br>pip install torch torchvision matplotlib seaborn scikit-learn</p>

###

<h2 align="left">Content</h2>

###

<p align="left">The notebook includes the following steps:<br><br>- Loading MNIST data : Using torchvision.datasets.MNIST.<br><br>- Data preprocessing: Normalization and tensor transformation.<br><br>- CNN model definition: Network architecture using PyTorch.<br><br>- Model training: Training loop and loss calculation.<br><br>-  Model evaluation: Calculating metrics (accuracy, F1-score, confusion matrix).<br><br>- Result visualization: Displaying images and predictions.</p>

###

<h2 align="left">Use</h2>

###

<p align="left">To run the notebook, open a terminal and run:<br>jupyter notebook Part1_Atelier2_DL.ipynb</p>

###

<h2 align="left">Training the CNN model:</h2>

###

<h4 align="left">5 epochs with different values</h4>

###

<p align="left">epoch 1 : 9 step with an accuracy of 93.52% and Loss: 0.2145</p>

###

<p align="left">epoch 2 : 9 step with an accuracy of 97.61% and Loss: 0.0829</p>

###

<p align="left">epoch 3 : 9 step with an accuracy of 98.25% and Loss: 0.0593</p>

###

<p align="left">epoch 4 : 9 step with an accuracy of 98.58% and Loss: 0.0485</p>

###

<p align="left">epoch 5 : 9 step with an accuracy of 98.69% and Loss: 0.0425</p>

###

<h2 align="left">Confusion Matrix</h2>

###

![image](https://github.com/user-attachments/assets/578a8a91-d287-449c-9bd2-8dfce092d4be)

###

<p align="left">This confusion matrix evaluates the performance of a classification model on a dataset of handwritten digits (0-9). Values ​​on the diagonal (e.g., 1002 for 0, 986 for 1, 1017 for 7) show the correct predictions, indicating good overall accuracy. However, errors exist, such as confusions between 4 and 9 or 7 and 8, suggesting visual similarities that need improvement.</p>

###

<h2 align="left">Training Performance</h2>

###

![image](https://github.com/user-attachments/assets/8810c9ce-b53a-4dc1-844f-b9c4b58e9228)

###

<p align="left">- Loss per Epoch : La perte (Training Loss) diminue rapidement de 0.200 à environ 0.050 sur 4 époques, indiquant une bonne convergence du modèle.</p>

###

<p align="left">- Accuracy per Epoch : La précision (Training Accuracy) augmente de 94% à 98% en 4 époques, montrant une amélioration constante des prédictions.</p>

###
## Results

| Model       | Accuracy | F1 Score | Loss   | Execution Time (s) |
|-------------|----------|----------|--------|--------------------|
| CNN         | 98.76%   | 0.9876   | 0.0412 | 45.32              |
| Faster R-CNN| 97.83%   | 0.9781   | 0.0698 | 112.45             |
| VGG16       | 99.21%   | 0.9920   | 0.0289 | 187.63             |
| AlexNet     | 98.94%   | 0.9893   | 0.0376 | 156.19             |
| ViT         | 98.52%   | 0.9851   | 0.0482 | 203.76             |

## Key Observations

- **Highest Accuracy**: VGG16 (99.21%)
- **Best F1 Score**: VGG16 (0.9920)
- **Lowest Loss**: VGG16 (0.0289)
- **Fastest Execution**: CNN (45.32s)
- **Transformer Performance**: ViT achieves competitive accuracy (98.52%) but with the longest runtime (203.76s).

## Conclusion

- **VGG16** remains the top performer in accuracy and F1 Score, though it is slower.  
- **CNN** offers the best speed-accuracy tradeoff for lightweight deployment.  
- **ViT** demonstrates the potential of transformer-based models but requires significantly more computational resources.  

*Prepared by Badr Benabdellah*
