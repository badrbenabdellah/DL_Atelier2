Ce projet est un notebook Jupyter implémentant un réseau de neurones convolutionnel (CNN) pour la classification d'images MNIST en utilisant PyTorch.

Prérequis

Assurez-vous d'avoir installé les bibliothèques nécessaires avant d'exécuter le notebook :

pip install torch torchvision matplotlib seaborn scikit-learn

Contenu

Le notebook comprend les étapes suivantes :

Chargement des données MNIST : Utilisation de torchvision.datasets.MNIST.

Prétraitement des données : Normalisation et transformation en tenseurs.

Définition du modèle CNN : Architecture du réseau avec PyTorch.

Entraînement du modèle : Boucle d'entraînement et calcul de la perte.

Évaluation du modèle : Calcul des métriques (accuracy, F1-score, matrice de confusion).

Visualisation des résultats : Affichage des images et des prédictions.

Utilisation

Pour exécuter le notebook, ouvrez un terminal et lancez :
jupyter notebook Part1_Atelier2_DL.ipynb
