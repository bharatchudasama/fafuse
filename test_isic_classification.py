import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# --- Import your custom model and dataloader ---
from lib.FAFuse_Classifier import FAFuse_Classifier
from utils.dataloader_classification import SkinClassificationDataset, get_transforms

def test(model, test_loader, device, num_classes):
    """
    Evaluates the trained classification model on the test set.
    """
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    print("\n#################### Start Testing ####################")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Calculate and Print Metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted') # Use 'weighted' for imbalanced classes
    
    print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Overall Weighted F1-Score: {f1:.4f}\n")

    # --- Classification Report ---
    # This shows precision, recall, and f1-score for each class
    class_names = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
    print("--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # --- Confusion Matrix ---
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Optional: Save a visualization of the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\nSaved confusion matrix plot to confusion_matrix.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FAFuse Classification Testing')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for testing')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
    parser.add_argument('--model_path', type=str, default='snapshots/FAFuse_Classifier/best_classifier.pth',
                        help='path to the saved best model checkpoint')
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    _, test_transform = get_transforms() # We only need the validation/test transform

    # --- Load Data ---
    test_csv_path = 'test_labels.csv'
    if not os.path.exists(test_csv_path):
        print(f"Error: {test_csv_path} not found. Please run the label preparation script first.")
        exit()
        
    test_dataset = SkinClassificationDataset(csv_file=test_csv_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Loaded {len(test_dataset)} images for testing.")

    # --- Load Model ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        exit()
        
    model = FAFuse_Classifier(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    print(f"Successfully loaded model from {args.model_path}")

    # --- Run Inference ---
    test(model, test_loader, device, args.num_classes)
