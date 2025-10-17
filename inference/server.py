

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
import nibabel as nib
import io
from pathlib import Path
import tempfile
import uuid
from datetime import datetime
import logging

from models.unet3d_monai import get_model

# Import custom modules
# from models.unet import get_model
# from inference.predictor import BraTSPredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BraTS Glioma Segmentation API",
    description="API for brain tumor segmentation using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
DEVICE = None
PREDICTOR = None
UPLOAD_DIR = Path("./uploads")
RESULTS_DIR = Path("./results")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction_id: str
    status: str
    message: str
    metrics: Optional[dict] = None
    download_url: Optional[str] = None
    timestamp: str


class BraTSPredictor:
    """Predictor class for inference"""
    
    def __init__(self, model, device, patch_size=(128, 128, 128)):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.model.eval()
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize MRI volume"""
        mask = volume > 0
        if not mask.any():
            return volume
        
        mean = np.mean(volume[mask])
        std = np.std(volume[mask])
        
        if std > 0:
            volume_norm = volume.copy()
            volume_norm[mask] = (volume[mask] - mean) / std
            return volume_norm
        return volume
    
    def preprocess(self, modalities: dict) -> torch.Tensor:
        """
        Preprocess input modalities
        
        Args:
            modalities: Dict with keys ['T1', 'T1Gd', 'T2', 'FLAIR']
        
        Returns:
            Preprocessed tensor (1, 4, D, H, W)
        """
        # Stack and normalize modalities
        volumes = []
        for mod in ['T1', 'T1Gd', 'T2', 'FLAIR']:
            if mod not in modalities:
                raise ValueError(f"Missing modality: {mod}")
            
            volume = modalities[mod]
            volume_norm = self.normalize_volume(volume)
            volumes.append(volume_norm)
        
        # Stack along channel dimension
        image = np.stack(volumes, axis=0)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        
        return image_tensor
    
    @torch.no_grad()
    def predict_sliding_window(self, image: torch.Tensor) -> np.ndarray:
        """
        Predict using sliding window approach for full volume
        
        Args:
            image: Input tensor (1, 4, D, H, W)
        
        Returns:
            Segmentation mask (D, H, W)
        """
        _, _, D, H, W = image.shape
        pd, ph, pw = self.patch_size
        
        # Initialize output volume
        output = torch.zeros(1, 5, D, H, W, device=self.device)  # 5 classes
        count = torch.zeros(1, 1, D, H, W, device=self.device)
        
        # Calculate stride (50% overlap)
        stride = [pd // 2, ph // 2, pw // 2]
        
        # Sliding window
        for d in range(0, D - pd + 1, stride[0]):
            for h in range(0, H - ph + 1, stride[1]):
                for w in range(0, W - pw + 1, stride[2]):
                    # Extract patch
                    patch = image[:, :, d:d+pd, h:h+ph, w:w+pw].to(self.device)
                    
                    # Predict
                    pred = self.model(patch)
                    
                    # Add to output
                    output[:, :, d:d+pd, h:h+ph, w:w+pw] += pred
                    count[:, :, d:d+pd, h:h+ph, w:w+pw] += 1
        
        # Average overlapping predictions
        output = output / count
        
        # Get final segmentation
        segmentation = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        return segmentation.astype(np.uint8)
    
    @torch.no_grad()
    def predict(self, modalities: dict) -> np.ndarray:
        """
        Main prediction function
        
        Args:
            modalities: Dict with MRI modalities
        
        Returns:
            Segmentation mask
        """
        # Preprocess
        image = self.preprocess(modalities)
        
        # Predict
        if image.shape[2:] == self.patch_size:
            # Single patch prediction
            image = image.to(self.device)
            output = self.model(image)
            segmentation = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        else:
            # Sliding window for large volumes
            segmentation = self.predict_sliding_window(image)
        
        return segmentation


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Build model
    model = get_model(
        model_name=config['model']['name'],
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        base_channels=config['model']['base_channels'],
        depth=config['model'].get('depth', 4)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, config


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global MODEL, DEVICE, PREDICTOR
    
    # Set device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")
    
    # Load model
    checkpoint_path = "./checkpoints/best_checkpoint.pth"  # Update this path
    MODEL, config = load_model(checkpoint_path, DEVICE)
    
    # Initialize predictor
    patch_size = tuple(config['inference']['patch_size'])
    PREDICTOR = BraTSPredictor(MODEL, DEVICE, patch_size=patch_size)
    
    logger.info("Server ready for predictions")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BraTS Glioma Segmentation API",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    t1n: UploadFile = File(..., description="T1 native MRI"),
    t1c: UploadFile = File(..., description="T1 post-contrast MRI"),
    t2w: UploadFile = File(..., description="T2-weighted MRI"),
    t2f: UploadFile = File(..., description="T2-FLAIR MRI")
):
    """
    Predict tumor segmentation from MRI scans
    
    Args:
        t1n: T1 native NIfTI file
        t1c: T1 post-contrast NIfTI file
        t2w: T2-weighted NIfTI file
        t2f: T2-FLAIR NIfTI file
    
    Returns:
        Prediction response with download link
    """
    try:
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        logger.info(f"Processing prediction {prediction_id}")
        
        # Load NIfTI files
        modalities = {}
        modality_names = {
            't1n': 'T1',
            't1c': 'T1Gd',
            't2w': 'T2',
            't2f': 'FLAIR'
        }
        
        uploaded_files = {
            't1n': t1n,
            't1c': t1c,
            't2w': t2w,
            't2f': t2f
        }
        
        nifti_header = None
        
        for key, file in uploaded_files.items():
            # Read file content
            content = await file.read()
            
            # Load NIfTI
            nifti_img = nib.Nifti1Image.from_bytes(content)
            
            # Store header from first file
            if nifti_header is None:
                nifti_header = nifti_img.header
                affine = nifti_img.affine
            
            # Get data
            data = nifti_img.get_fdata()
            modalities[modality_names[key]] = data
        
        # Run prediction
        logger.info("Running model inference...")
        segmentation = PREDICTOR.predict(modalities)
        
        # Save segmentation as NIfTI
        output_path = RESULTS_DIR / f"{prediction_id}_segmentation.nii.gz"
        seg_nifti = nib.Nifti1Image(segmentation, affine, nifti_header)
        nib.save(seg_nifti, str(output_path))
        
        logger.info(f"Prediction saved to {output_path}")
        
        # Calculate basic statistics
        unique, counts = np.unique(segmentation, return_counts=True)
        label_counts = dict(zip(unique.tolist(), counts.tolist()))
        
        # Calculate tumor volumes (assuming 1mm³ voxels)
        total_voxels = segmentation.size
        tumor_voxels = np.sum(segmentation > 0)
        tumor_volume_ml = tumor_voxels / 1000  # Convert to mL
        
        metrics = {
            "label_counts": label_counts,
            "total_voxels": int(total_voxels),
            "tumor_voxels": int(tumor_voxels),
            "tumor_volume_ml": float(tumor_volume_ml),
            "tumor_percentage": float(tumor_voxels / total_voxels * 100)
        }
        
        return PredictionResponse(
            prediction_id=prediction_id,
            status="success",
            message="Segmentation completed successfully",
            metrics=metrics,
            download_url=f"/download/{prediction_id}",
            timestamp=timestamp
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{prediction_id}")
async def download_segmentation(prediction_id: str):
    """
    Download segmentation result
    
    Args:
        prediction_id: Prediction ID
    
    Returns:
        NIfTI file
    """
    file_path = RESULTS_DIR / f"{prediction_id}_segmentation.nii.gz"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return FileResponse(
        path=file_path,
        filename=f"segmentation_{prediction_id}.nii.gz",
        media_type="application/gzip"
    )


'''if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 
    '''
