import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# Weak augmentation
weak_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    transforms.ToTensor(),
    # Uncomment the normalization if needed
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Strong augmentation
strong_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    transforms.RandAugment(),
    transforms.ToTensor(),
    # Uncomment the normalization if needed
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test-time transformations (no augmentations, just resizing)
test_transform = transforms.Compose([
    transforms.Resize([224, 224], interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    # Uncomment the normalization if needed
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
