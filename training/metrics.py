"""
Loss Functions and Evaluation Metrics for BraTS Glioma Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation
    """
    
    def __init__(self, smooth: float = 1e-5, include_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, D, H, W) - logits
            targets: (B, D, H, W) - class indices
        """
        # Apply softmax to get probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Flatten spatial dimensions
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), 
                                               targets_one_hot.size(1), -1)
        
        # Calculate Dice coefficient for each class
        intersection = (predictions * targets_one_hot).sum(dim=2)
        cardinality = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Optionally exclude background
        if not self.include_background:
            dice_score = dice_score[:, 1:]
        
        # Return mean Dice loss
        dice_loss = 1 - dice_score.mean()
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, D, H, W) - logits
            targets: (B, D, H, W) - class indices
        """
        # Calculate cross entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Get predicted probabilities
        p = F.softmax(predictions, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice + Cross Entropy Loss
    """
    
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        return self.dice_weight * dice + self.ce_weight * ce


class DiceScore:
    """
    Lesion-wise Dice Similarity Coefficient
    """
    
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray, 
                 class_idx: int = None) -> float:
        """
        Calculate Dice score
        
        Args:
            predictions: Binary or multi-class segmentation
            targets: Ground truth segmentation
            class_idx: Specific class to evaluate (None for all)
        
        Returns:
            Dice score
        """
        if class_idx is not None:
            pred_binary = (predictions == class_idx).astype(np.uint8)
            target_binary = (targets == class_idx).astype(np.uint8)
        else:
            pred_binary = (predictions > 0).astype(np.uint8)
            target_binary = (targets > 0).astype(np.uint8)
        
        # Calculate Dice
        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary)
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        dice = (2.0 * intersection) / union
        return dice


class Hausdorff95:
    """
    95% Hausdorff Distance
    """
    
    def __init__(self):
        pass
    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray,
                 class_idx: int = None) -> float:
        """
        Calculate 95% Hausdorff distance
        
        Args:
            predictions: Binary or multi-class segmentation
            targets: Ground truth segmentation
            class_idx: Specific class to evaluate
        
        Returns:
            95% Hausdorff distance in mm
        """
        if class_idx is not None:
            pred_binary = (predictions == class_idx).astype(np.uint8)
            target_binary = (targets == class_idx).astype(np.uint8)
        else:
            pred_binary = (predictions > 0).astype(np.uint8)
            target_binary = (targets > 0).astype(np.uint8)
        
        # Check if either mask is empty
        if not pred_binary.any() or not target_binary.any():
            if not pred_binary.any() and not target_binary.any():
                return 0.0
            else:
                return float('inf')
        
        # Get surface points
        pred_surface = self._get_surface_points(pred_binary)
        target_surface = self._get_surface_points(target_binary)
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            return float('inf')
        
        # Calculate distances
        distances_pred_to_target = self._compute_distances(pred_surface, target_surface)
        distances_target_to_pred = self._compute_distances(target_surface, pred_surface)
        
        # Combine distances
        all_distances = np.concatenate([distances_pred_to_target, 
                                       distances_target_to_pred])
        
        # Return 95th percentile
        hd95 = np.percentile(all_distances, 95)
        return hd95
    
    def _get_surface_points(self, binary_mask: np.ndarray) -> np.ndarray:
        """Extract surface points from binary mask"""
        # Apply erosion to get surface
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(binary_mask)
        surface = binary_mask ^ eroded
        
        # Get coordinates of surface points
        surface_points = np.argwhere(surface)
        return surface_points
    
    def _compute_distances(self, points1: np.ndarray, 
                          points2: np.ndarray) -> np.ndarray:
        """Compute minimum distances from points1 to points2"""
        distances = []
        for point in points1:
            dists = np.sqrt(np.sum((points2 - point) ** 2, axis=1))
            distances.append(np.min(dists))
        return np.array(distances)


class BraTSMetrics:
    """
    Complete metrics calculator for BraTS
    """
    
    def __init__(self):
        self.dice_score = DiceScore()
        self.hausdorff_95 = Hausdorff95()
        
        # Define region mappings
        self.regions = {
            'ET': 3,           # Enhancing Tumor
            'NETC': 1,         # Non-enhancing Tumor Core
            'SNFH': 2,         # Surrounding Non-enhancing FLAIR Hyperintensity
            'RC': 4,           # Resection Cavity
            'TC': [1, 3],      # Tumor Core (NETC + ET)
            'WT': [1, 2, 3]    # Whole Tumor (NETC + SNFH + ET)
        }
    
    def calculate_metrics(self, predictions: np.ndarray, 
                         targets: np.ndarray) -> dict:
        """
        Calculate all metrics for BraTS evaluation
        
        Args:
            predictions: Predicted segmentation (H, W, D)
            targets: Ground truth segmentation (H, W, D)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for region_name, region_labels in self.regions.items():
            # Create binary masks for region
            if isinstance(region_labels, list):
                pred_mask = np.isin(predictions, region_labels).astype(np.uint8)
                target_mask = np.isin(targets, region_labels).astype(np.uint8)
            else:
                pred_mask = (predictions == region_labels).astype(np.uint8)
                target_mask = (targets == region_labels).astype(np.uint8)
            
            # Calculate Dice score
            dice = self.dice_score(pred_mask, target_mask)
            metrics[f'{region_name}_Dice'] = dice
            
            # Calculate Hausdorff distance
            hd95 = self.hausdorff_95(pred_mask, target_mask)
            metrics[f'{region_name}_HD95'] = hd95
        
        return metrics
    
    def aggregate_metrics(self, all_metrics: list) -> dict:
        """
        Aggregate metrics across multiple cases
        
        Args:
            all_metrics: List of metric dictionaries
        
        Returns:
            Aggregated metrics (mean and std)
        """
        aggregated = {}
        
        # Get all metric names
        metric_names = all_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics 
                     if not np.isinf(m[metric_name])]
            
            if len(values) > 0:
                aggregated[f'{metric_name}_mean'] = np.mean(values)
                aggregated[f'{metric_name}_std'] = np.std(values)
            else:
                aggregated[f'{metric_name}_mean'] = float('nan')
                aggregated[f'{metric_name}_std'] = float('nan')
        
        return aggregated


if __name__ == '__main__':
    # Test loss functions
    batch_size = 2
    num_classes = 5
    depth, height, width = 128, 128, 128
    
    # Create dummy data
    predictions = torch.randn(batch_size, num_classes, depth, height, width)
    targets = torch.randint(0, num_classes, (batch_size, depth, height, width))
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    loss_dice = dice_loss(predictions, targets)
    print(f"Dice Loss: {loss_dice.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss()
    loss_combined = combined_loss(predictions, targets)
    print(f"Combined Loss: {loss_combined.item():.4f}")
    
    # Test metrics
    pred_np = torch.argmax(predictions, dim=1)[0].cpu().numpy()
    target_np = targets[0].cpu().numpy()
    
    metrics_calc = BraTSMetrics()
    metrics = metrics_calc.calculate_metrics(pred_np, target_np)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")