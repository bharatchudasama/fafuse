# import argparse
# import os  # <-- FIX: Added the missing import
# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from datetime import datetime
# from sklearn.metrics import f1_score, accuracy_score

# # --- Import the new Classification Model and Dataloader ---
# from lib.FAFuse_Classifier import FAFuse_Classifier
# from utils.dataloader_classification import SkinClassificationDataset, get_transforms

# def validate(model, dataloader, criterion):
#     """
#     Evaluates the model on the validation set.
#     Returns validation loss, accuracy, and F1-score.
#     """
#     model.eval()
#     all_preds = []
#     all_labels = []
#     total_loss = 0.0

#     with torch.no_grad():
#         for images, labels in dataloader:
#             images = Variable(images).cuda()
#             labels = Variable(labels).cuda()

#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()

#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     avg_loss = total_loss / len(dataloader)
#     accuracy = accuracy_score(all_labels, all_preds)
#     f1 = f1_score(all_labels, all_preds, average='macro') # 'macro' for multi-class

#     return avg_loss, accuracy, f1


# def train():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epoch', type=int, default=50, help='epoch number')
#     parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
#     parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
#     parser.add_argument('--train_save', type=str, default='FAFuse_Classifier')
#     # --- FIX: Changed default to 7 for multi-class classification ---
#     parser.add_argument('--num_classes', type=int, default=7, help='output channel of network')
#     opt = parser.parse_args()

#     # ---- build models ----
#     model = FAFuse_Classifier(num_classes=opt.num_classes, pretrained=True).cuda()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), opt.lr)

#     # ---- data prepare ----
#     train_transform, val_transform = get_transforms()
#     train_dataset = SkinClassificationDataset(csv_file='train_labels.csv', transform=train_transform)
#     train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=4)

#     val_dataset = SkinClassificationDataset(csv_file='val_labels.csv', transform=val_transform)
#     val_loader = DataLoader(val_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=4)
    
#     total_step = len(train_loader)
    
#     print("#"*20, "Start Training", "#"*20)

#     best_accuracy = 0.0
    
#     # --- Create save directory ---
#     save_path = f'snapshots/{opt.train_save}/'
#     os.makedirs(save_path, exist_ok=True) # This line needs 'os' to be imported

#     for epoch in range(1, opt.epoch + 1):
#         model.train()
#         epoch_loss = 0.0
        
#         for i, (images, labels) in enumerate(train_loader, start=1):
#             images = Variable(images).cuda()
#             labels = Variable(labels).cuda()

#             # ---- forward ----
#             outputs = model(images)
#             loss = criterion(outputs, labels)
            
#             # ---- backward ----
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()

#             # ---- train visualization ----
#             if i % 20 == 0 or i == total_step:
#                 print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
#                       format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))

#         # --- Validation after each epoch ---
#         val_loss, val_acc, val_f1 = validate(model, val_loader, criterion)
#         print(f"\nEpoch {epoch} Validation -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1-Score: {val_f1:.4f}\n")

#         # --- Save the best model ---
#         if val_acc > best_accuracy:
#             best_accuracy = val_acc
#             torch.save(model.state_dict(), os.path.join(save_path, 'best_classifier.pth'))
#             print(f'[Saving Snapshot:] New best model saved with accuracy: {best_accuracy:.4f}\n')

# if __name__ == '__main__':
#     train()
 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
import argparse
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm

# --- Import your custom model and dataloader ---
from lib.FAFuse_Classifier import FAFuse_Classifier
from utils.dataloader_classification import SkinClassificationDataset, get_transforms

def train():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = 'snapshots/FAFuse_Classifier/'
    os.makedirs(save_path, exist_ok=True)

    # --- Load Data ---
    train_transform, val_transform = get_transforms()
    train_dataset = SkinClassificationDataset(csv_file='train_labels.csv', transform=train_transform)
    val_dataset = SkinClassificationDataset(csv_file='val_labels.csv', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- STRATEGY 1: CALCULATE CLASS WEIGHTS FOR WEIGHTED LOSS ---
    print("Calculating class weights to handle data imbalance...")
    train_labels_df = pd.read_csv('train_labels.csv')
    class_counts = train_labels_df['label'].value_counts().sort_index().values
    total_samples = class_counts.sum()
    class_weights = torch.tensor([total_samples / c for c in class_counts], dtype=torch.float32).to(device)
    print("Class Weights:", class_weights)
    # ----------------------------------------------------------------

    # --- Model, Loss, and Optimizer ---
    model = FAFuse_Classifier(num_classes=args.num_classes, pretrained=True).to(device)
    
    # --- Pass the calculated weights to the loss function ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training Loop ---
    best_val_accuracy = 0.0
    print("\n#################### Start Training ####################")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for a nice progress bar
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")

        for i, (images, labels) in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Update progress bar description
            train_pbar.set_postfix({'Loss': running_loss / (i + 1)})

        # --- Validation Step ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"\nEpoch {epoch}/{args.epochs} -> "
              f"Validation Accuracy: {val_accuracy*100:.2f}%, "
              f"Validation F1-Score: {val_f1:.4f}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(save_path, 'best_classifier.pth'))
            print(f"----> New best model saved with accuracy: {best_val_accuracy*100:.2f}%")
        
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FAFuse Classification Training')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs') # Increased epochs
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_classes', type=int, default=7, help='output channel of network')
    args = parser.parse_args()

    train()
