import torch
import numpy as np
from unet import UNet
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import get_train_test_datasets

# Parameters setting
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
annotation_file = "annotations.json"
model_path = "models/sea_turtle_unet_best.pth"
pred_img_dir = "pred_imgs/5"

# Load model
model = UNet(in_channels=1, out_channels=3).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set to evaluation mode

train_dataset, val_dataset, test_dataset = get_train_test_datasets()

def predict_mask(image, model):
    """
    Predict the mask for a single image
    :param image: Preprocessed image tensor
    :param model: Trained model
    :return: Predicted mask in numpy array format
    """
    # Preprocess
    image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        pred_masks = torch.sigmoid(output)        # Convert logits to probabilities
        pred_masks = (pred_masks > 0.5).float()   # Binarize
        pred_masks = pred_masks.cpu().numpy()[0]  # Remove batch dimension
    
    return pred_masks

def visualize_prediction(image, masks, save_path):
    """
    Visualize the prediction results
    """
    # Set matplotlib style
    plt.style.use('default')

    # Create a new image
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Display the original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Image')
    axes[0, 0].axis('off')
    
    # Display each part's mask
    parts = ['Head', 'Flippers', 'Turtle']
    for i in range(3):
        row = (i + 1) // 2
        col = (i + 1) % 2
        axes[row, col].imshow(masks[i], cmap='gray')
        axes[row, col].set_title(parts[i])
        axes[row, col].axis('off')
    
    plt.tight_layout()  # Adjust the spacing between subplots
    plt.savefig(save_path, bbox_inches='tight', dpi=100) # Save the image
    plt.close(fig)      # Clean up

def test_model(num_samples=10):
    """
    Test the model on a specified number of images
    :param num_samples: Number of images to test (default: 10)
    """
    
    # Get a random sample of image IDs
    selected_ids = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    print(f"Testing model on {len(selected_ids)} images...")
    
    for img_id in tqdm(selected_ids):
        # Load the image
        image, true_masks = test_dataset[img_id]

        # Convert image format
        image_np = image.squeeze().numpy()  # Remove channel dimension
        image_np = (image_np * 0.224 + 0.456) * 255  # Inverse normalization using standard deviation and mean
        image_np = image_np.astype(np.uint8)
        
        # Predict the mask
        pred_masks = predict_mask(image, model)
        
        # Save the results
        save_path = f"{pred_img_dir}/{img_id}_pred.png"
        visualize_prediction(image_np, pred_masks, save_path)

        # Calculate and print IoU scores
        parts = ['Head', 'Flippers', 'Turtle']
        true_masks_np = true_masks.numpy()
        for i, part in enumerate(parts):
            iou = calculate_iou(pred_masks[i], true_masks_np[i])
            print(f"IoU for {part}: {iou:.4f}")
        print("-" * 50)

def calculate_iou(pred_mask, true_mask):
    """
    Calculate IoU (Intersection over Union)
    """
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    if np.sum(union) == 0:
        return 0.0
    iou = np.sum(intersection) / np.sum(union)
    return iou

def evaluate_model():
    """
    Evaluate the model on the entire test set
    """
    _, _, test_dataset = get_train_test_datasets()
    
    ious = {
        'head': [], 'flippers': [], 'turtle': []
    }
    
    print("Evaluating model on entire test set...")
    
    for idx in tqdm(range(len(test_dataset))):
        image, true_masks = test_dataset[idx]
        
        # Predict mask
        pred_masks = predict_mask(image, model)
        
        # Calculate IoU for each part
        true_masks_np = true_masks.numpy()
        for i, part in enumerate(ious.keys()):
            iou = calculate_iou(pred_masks[i], true_masks_np[i])
            ious[part].append(iou)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    for part, scores in ious.items():
        mean_iou = np.mean(scores)
        print(f"Mean IoU for {part}: {mean_iou:.4f}")

if __name__ == "__main__":
    # Create the directory to save results
    import os
    os.makedirs(pred_img_dir, exist_ok=True)
    
    # Run both test and evaluation
    print("Running tests on sample images...")
    test_model(num_samples=10)
    
    print("\nRunning full evaluation...")
    evaluate_model()
