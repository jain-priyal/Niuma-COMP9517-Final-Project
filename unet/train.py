import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet import UNet
from dataset import get_train_test_datasets
from tqdm import tqdm

# Parameters setting
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(device)
num_epochs = 100          # Max epochs
patience = 5             # Early stopping patience
learning_rate = 1e-3     # Learning rate
min_delta = 1e-4        # Minimum change to qualify as an improvement
batch_size = 16          # Batch size

# Paths
checkpoint_path = "models/sea_turtle_unet.pth"
best_model_path = "models/sea_turtle_unet_best.pth"

def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, best_val_loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss
    }, save_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    return start_epoch, loss

def validate(model, val_loader, criterion, device):
    """Validating"""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train():
    # Initialize dataset and data loader
    train_dataset, val_dataset, test_dataset = get_train_test_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss function and optimizer
    model = UNet(in_channels=1, out_channels=3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=3, 
        verbose=True
    )

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        start_epoch, loss = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for image, true_masks in progress_bar:
            image, true_masks = image.to(device), true_masks.to(device)

            # Forward propagation
            pred_masks = model(image)
            loss = criterion(pred_masks, true_masks)
            epoch_train_loss += loss.item()

            # Backward propagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation phase
        val_loss = validate(model, val_loader, criterion, device)

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save model after each epoch
        save_checkpoint(epoch, model, optimizer, avg_train_loss, val_loss, best_val_loss, checkpoint_path)

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model!")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    print("Training completed, model saved.")
    return model

if __name__ == "__main__":
    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train model
    model = train()
