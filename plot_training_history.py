# import pandas as pd
# import matplotlib.pyplot as plt
# import argparse
# import os
# import re
# import numpy as np

# def plot_fafuse_history(log_dir, output_dir):
#     """
#     Plots the training and validation history for the FAFuse classification model.
#     Assumes train_log.csv and val_log.csv exist in the log_dir.
#     """
#     print(f"--- Plotting FAFuse History from '{log_dir}' ---")
#     train_log_path = os.path.join(log_dir, 'train_log.csv')
#     val_log_path = os.path.join(log_dir, 'val_log.csv')

#     if not os.path.exists(train_log_path) or not os.path.exists(val_log_path):
#         print(f"Error: Could not find log files in {log_dir}. Please check the path.")
#         return

#     # Read the data
#     train_df = pd.read_csv(train_log_path)
#     val_df = pd.read_csv(val_log_path)

#     os.makedirs(output_dir, exist_ok=True)

#     # --- Plot 1: Training Loss vs. Validation Accuracy/F1 ---
#     fig, ax1 = plt.subplots(figsize=(12, 6))

#     color = 'royalblue'
#     ax1.set_xlabel('Epoch', fontsize=12)
#     ax1.set_ylabel('Training Loss', color=color, fontsize=12)
#     ax1.plot(train_df['epoch'], train_df['loss'], label='Training Loss', color=color, linewidth=2)
#     ax1.tick_params(axis='y', labelcolor=color)
#     ax1.grid(True, linestyle='--', alpha=0.6)

#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#     color = 'forestgreen'
#     ax2.set_ylabel('Validation Score', color=color, fontsize=12)
#     ax2.plot(val_df['epoch'], val_df['accuracy'], label='Validation Accuracy', color=color, marker='o', linestyle='--')
#     ax2.plot(val_df['epoch'], val_df['f1_score'], label='Validation F1-Score', color='darkorange', marker='s', linestyle='--')
#     ax2.tick_params(axis='y', labelcolor=color)

#     fig.suptitle('FAFuse: Training Loss vs. Validation Metrics', fontsize=16, fontweight='bold')
#     fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=12)
#     fig.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout to make room for legend
    
#     plot_path = os.path.join(output_dir, 'fafuse_training_history.png')
#     plt.savefig(plot_path)
#     print(f"Saved FAFuse training history plot to {plot_path}")
#     plt.close()


# def plot_onet_history(log_dir, output_dir):
#     """
#     Parses the O-Net log.txt file and plots the training and validation loss.
#     """
#     print(f"--- Plotting O-Net History from '{log_dir}' ---")
#     log_file_path = os.path.join(log_dir, 'log.txt')

#     if not os.path.exists(log_file_path):
#         print(f"Error: Could not find log.txt in {log_dir}. Please check the path.")
#         return

#     # Regex patterns to find the data
#     iter_pattern = re.compile(r'(\d+) iterations per epoch')
#     train_loss_pattern = re.compile(r'iteration \d+ : loss : (\d+\.\d+)')
#     val_loss_pattern = re.compile(r'Validation --- Epoch: \d+, Dice Loss: (\d+\.\d+)')

#     # Extract data from log file
#     iters_per_epoch = None
#     train_losses = []
#     val_losses = []

#     with open(log_file_path, 'r') as f:
#         for line in f:
#             if not iters_per_epoch:
#                 match = iter_pattern.search(line)
#                 if match:
#                     iters_per_epoch = int(match.group(1))
#                     print(f"Found {iters_per_epoch} iterations per epoch in log file.")
            
#             train_match = train_loss_pattern.search(line)
#             if train_match:
#                 train_losses.append(float(train_match.group(1)))

#             val_match = val_loss_pattern.search(line)
#             if val_match:
#                 val_losses.append(float(val_match.group(1)))
    
#     if not iters_per_epoch or not train_losses or not val_losses:
#         print("Error: Could not parse necessary data from the log file.")
#         return

#     # Aggregate training losses by epoch
#     num_epochs_val = len(val_losses)
#     avg_train_loss_per_epoch = []
#     for i in range(num_epochs_val):
#         start_idx = i * iters_per_epoch
#         end_idx = start_idx + iters_per_epoch
#         epoch_losses = train_losses[start_idx:end_idx]
#         if epoch_losses:
#             avg_train_loss_per_epoch.append(np.mean(epoch_losses))
#         else:
#             # Handle case where training might have been interrupted
#             if len(avg_train_loss_per_epoch) > 0:
#                  avg_train_loss_per_epoch.append(avg_train_loss_per_epoch[-1]) # repeat last value
#             else:
#                  avg_train_loss_per_epoch.append(0)


#     epochs = range(1, num_epochs_val + 1)
#     os.makedirs(output_dir, exist_ok=True)

#     # --- Plotting Training and Validation Loss ---
#     plt.figure(figsize=(12, 6))
#     plt.plot(epochs, avg_train_loss_per_epoch, label='Avg. Training Loss', color='royalblue', linewidth=2)
#     plt.plot(epochs, val_losses, label='Validation Dice Loss', color='darkorange', marker='s', linestyle='--')
#     plt.title('O-Net: Training vs. Validation Loss over Epochs', fontsize=16, fontweight='bold')
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Loss', fontsize=12)
#     plt.legend(fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plot_path = os.path.join(output_dir, 'onet_loss_history.png')
#     plt.savefig(plot_path)
#     print(f"Saved O-Net loss history plot to {plot_path}")
#     plt.close()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Plot training history from log files.")
#     parser.add_argument('--model', type=str, required=True, choices=['fafuse', 'onet'],
#                         help="The model type ('fafuse' or 'onet') to plot history for.")
#     parser.add_argument('--log_dir', type=str, required=True,
#                         help="The directory containing the log files (e.g., './snapshots/FAFuse_Classifier/').")
#     parser.add_argument('--output_dir', type=str, default='./plots',
#                         help="The directory where the output plots will be saved.")
    
#     args = parser.parse_args()

#     if args.model == 'fafuse':
#         plot_fafuse_history(args.log_dir, args.output_dir)
#     elif args.model == 'onet':
#         plot_onet_history(args.log_dir, args.output_dir)
#     else:
#         print(f"Error: Unknown model type '{args.model}'")

import matplotlib.pyplot as plt
import numpy as np

# Epochs 1–30
epochs = list(range(1, 31))

# Known validation losses from logs
loss_dict = {
    1: 1.5219,
    2: 1.5869,
    3: 1.7193,
    9: 1.7669
}

# Create array with None for missing epochs
losses = [loss_dict.get(e, None) for e in epochs]

# --- Option 1: Raw losses with gaps ---
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='--', color='b', label='Recorded Losses')

# --- Option 2: Smoothed (interpolated) losses ---
valid_epochs = [e for e, l in zip(epochs, losses) if l is not None]
valid_losses = [l for l in losses if l is not None]

# Interpolate missing epochs linearly
interp_losses = np.interp(epochs, valid_epochs, valid_losses)
plt.plot(epochs, interp_losses, linestyle='-', color='orange', label='Interpolated Loss')

# Highlight best loss point
best_epoch = valid_epochs[valid_losses.index(max(valid_losses))]
best_loss = max(valid_losses)
plt.scatter(best_epoch, best_loss, color='r', s=100, label=f'Best Loss (Epoch {best_epoch})')

plt.title("Validation Loss over Epochs", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
