import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# Paths
annotation_file = "annotations.json"
img_dir = "."

class SeaTurtleDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, img_ids=None):
        """
        Initialize COCO dataset
        :param annotation_file: Path to COCO format annotation file
        :param img_dir: Directory containing image files
        :param transform: Data augmentation transformations
        :param img_ids: Optional list of image IDs to use (for train/test split)
        """
        self.coco = COCO(annotation_file)
        self.img_dir = img_dir
        self.transform = transform if transform else A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
        self.img_ids = img_ids if img_ids is not None else list(self.coco.imgs.keys())
        self.num_classes = len(self.coco.getCatIds())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # Get image ID and path
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = f"{self.img_dir}/{img_info['file_name']}"
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = image[..., None]    # (H, W, 1)

        # Create a mask with 3 channels
        # [head, flippers, turtle]
        num_parts = 3
        mask = np.zeros((num_parts, image.shape[0], image.shape[1]), dtype=np.float32)

        # Get annotations
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        
        for ann in annotations:
            cat_id = ann['category_id']
            binary_mask = self.coco.annToMask(ann)

            # Ensure binary_mask has same shape as image
            if binary_mask.shape != (image.shape[0], image.shape[1]):
                binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
                print(f"Resized binary mask from {binary_mask.shape} to {image.shape}")
            
            if cat_id == 1:    # turtle
                mask[2] = np.logical_or(mask[2], binary_mask)
            elif cat_id == 2:  # flipper
                mask[1] = np.logical_or(mask[1], binary_mask)
            elif cat_id == 3:  # head
                mask[0] = np.logical_or(mask[0], binary_mask)

        # CHW to HWC
        mask = np.transpose(mask, (1, 2, 0))
        
        # Transform image and mask
        try:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        except Exception as e:
            print(f"Error during transform for image {img_path}")
            print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
            raise e

        # HWC to CHW
        mask = mask.permute(2, 0, 1)
        
        return image, mask
    
def get_train_test_datasets(
    annotation_file=annotation_file,
    img_dir=img_dir,
    sample_ratio=1,  # Ratio of dataset to use
    val_size=0.1,
    test_size=0.2,
    random_state=42,
    train_transform=None,
    val_transform=None,
    test_transform=None
):
    """
    Create training and testing datasets
    :param sample_ratio: Ratio of dataset to use (0-1)
    """
    # Load COCO dataset
    coco = COCO(annotation_file)
    all_img_ids = list(coco.imgs.keys())
    
    # Sample the entire dataset
    num_samples = int(len(all_img_ids) * sample_ratio)
    selected_img_ids = np.random.choice(
        all_img_ids, 
        size=num_samples, 
        replace=False
    )
    
    # Split into training and testing sets
    train_val_ids, test_ids = train_test_split(
        selected_img_ids,
        test_size=test_size,
        random_state=random_state
    )

    # Second split: separate validation set
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size/(1-test_size),  # 调整验证集比例
        random_state=random_state
    )
    
    print(f"Dataset size: {len(all_img_ids)}")
    print(f"Sampled dataset size: {len(selected_img_ids)}")
    print(f"Training set size: {len(train_ids)}")
    print(f"Validation set size: {len(val_ids)}")
    print(f"Testing set size: {len(test_ids)}")
    
    # Create default transformations
    if train_transform is None:
        train_transform = A.Compose([
            # A.RandomResizedCrop(height=224, width=224),
            A.Resize(224, 224),
            # A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.456], std=[0.224]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    if val_transform is None:
        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.456], std=[0.224]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    
    if test_transform is None:
        test_transform = val_transform
    
    # Create training and testing datasets
    train_dataset = SeaTurtleDataset(
        annotation_file=annotation_file,
        img_dir=img_dir,
        transform=train_transform,
        img_ids=train_ids
    )

    val_dataset = SeaTurtleDataset(
        annotation_file=annotation_file,
        img_dir=img_dir,
        transform=val_transform,
        img_ids=val_ids
    )
    
    test_dataset = SeaTurtleDataset(
        annotation_file=annotation_file,
        img_dir=img_dir,
        transform=test_transform,
        img_ids=test_ids
    )
    
    return train_dataset, val_dataset, test_dataset

# Example usage
if __name__ == "__main__":
    # Initialize dataset
    dataset = SeaTurtleDataset(
        annotation_file="annotations.json",
        img_dir="images",
        transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
    )

    # Load data
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(len(dataset))

    # Visualize the first 5 samples
    import matplotlib.pyplot as plt

    for i in range(1,5):
        image, mask = dataset[i]
        image_np = image.squeeze().numpy() * 0.224 + 0.456
        image_np = np.clip(image_np, 0, 1)
        
        mask_np = mask.numpy()
        
        plt.figure(figsize=(10, 5))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(image_np, cmap='gray')
        plt.title('Image')
        plt.axis('off')
        
        # Display mask
        aggregated_mask = np.sum(mask_np, axis=0)
        plt.subplot(1, 2, 2)
        plt.imshow(aggregated_mask, cmap='gray')
        plt.title('Aggregated Mask')
        plt.axis('off')
        
        plt.show()
