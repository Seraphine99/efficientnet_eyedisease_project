import matplotlib.pyplot as plt
import numpy as np

# Data manually extracted from your terminal output
epochs = np.arange(1, 21)
train_acc = [0.69, 0.75, 0.77, 0.79, 0.80, 0.80, 0.82, 0.83, 0.83, 0.83, 
             0.84, 0.84, 0.85, 0.85, 0.85, 0.85, 0.85, 0.86, 0.86, 0.87]

val_acc = [0.8592, 0.8310, 0.8239, 0.8263, 0.8427, 0.8451, 0.8521, 0.8521, 0.8568, 0.8638, 
           0.8615, 0.8662, 0.8685, 0.8709, 0.8685, 0.8685, 0.8685, 0.8732, 0.8685, 0.8685]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label='Training Accuracy', color='#1f77b4', marker='o', linewidth=2)
plt.plot(epochs, val_acc, label='Validation Accuracy', color='#ff7f0e', marker='s', linewidth=2)

plt.title('Phase 2: Fine-Tuning Accuracy (Surgical Unfreeze)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Highlight the final result
plt.annotate(f'Final Val: {val_acc[-1]:.2%}', 
             xy=(20, val_acc[-1]), xytext=(15, 0.75),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.savefig('learning_curve.png')
plt.show()
print("✅ Learning curve saved as 'learning_curve.png'")