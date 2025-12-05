"""
RS-Diffusion Inference Script
==============================

This script performs inference using a trained RS-Diffusion model to correct
rolling shutter distortions in images. It takes rolling shutter (RS) images
as input and generates corrected global shutter (GS) images along with optical
flow predictions.

Usage:
------
Basic usage with single image:
    python inference.py --input path/to/rolling_shutter_image.jpg --checkpoint checkpoint/RS_Real.pt

Process multiple images from a folder:
    python inference.py --input path/to/images_folder/ --checkpoint checkpoint/RS_Real.pt --output results/

With custom configuration:
    python inference.py --input path/to/image.jpg --config config/custom_config.yaml --checkpoint checkpoint/RS_Real.pt

Arguments:
----------
--input         : Path to input image or folder containing images (required)
--checkpoint    : Path to trained model checkpoint (required)
--config        : Path to configuration YAML file (optional, default: config/RS_real_config_teat.yaml)
--output        : Output directory for results (optional, default: inference_output/)
--device        : Device to run inference on (optional, default: cuda if available, else cpu)
--cond_scale    : Classifier-free guidance scale (optional, default: 1.0)
--rescaled_phi  : Rescaled phi for guidance (optional, default: 0.7)
--save_flow_viz : Save flow visualization images (optional, default: True)
--image_size    : Image size for processing (optional, default: from config or 64)

Output:
-------
The script generates the following outputs in the specified output directory:
- <image_name>_corrected.png  : Corrected global shutter image
- <image_name>_flow.npy       : Predicted optical flow (numpy array)
- <image_name>_flow_viz.png   : Flow visualization (if --save_flow_viz is True)
- <image_name>_comparison.png : Side-by-side comparison of input and output

Requirements:
-------------
- PyTorch
- torchvision
- numpy
- PIL
- yaml
- tqdm
- ema_pytorch
- accelerate
- einops

Example:
--------
python inference.py \\
    --input test-office/rgb/frame_0001.jpg \\
    --checkpoint checkpoint/RS_Real.pt \\
    --output my_results/ \\
    --cond_scale 1.0 \\
    --save_flow_viz True
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import os
import cv2
from torchvision import transforms as T
from torchvision import utils
from tqdm import tqdm

# Import model components
from model.diffusion_model import Unet, RS_Diffusion
from flow_utils import flow_to_image, flow_warp, upsample2d_flow_as


def load_model(config_path, checkpoint_path, device):
    """
    Load the trained RS-Diffusion model with EMA weights.
    
    Args:
        config_path (str): Path to configuration YAML file
        checkpoint_path (str): Path to model checkpoint
        device (str): Device to load model on ('cuda' or 'cpu')
    
    Returns:
        tuple: (model, config) - Loaded diffusion model with EMA and configuration dict
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Loading configuration from: {config_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Initialize the U-Net model
    model = Unet(
        dim=config["model"]["dim"],
        dim_mults=tuple(config["model"]["dim_mults"]),
        num_classes=config["model"]["num_classes"],
        cond_drop_prob=config["model"]["cond_drop_prob"],
    )
    
    # Initialize the diffusion model
    diffusion = RS_Diffusion(
        model,
        image_size=config["diffusion"]["image_size"],
        timesteps=config["diffusion"]["timesteps"],
        sampling_timesteps=config["diffusion"]["sampling_timesteps"],
        beta_schedule=config["diffusion"]["beta_schedule"],
        objective=config["diffusion"]["objective"],
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the EMA model state
    if 'ema' in checkpoint:
        from ema_pytorch import EMA
        ema = EMA(diffusion, beta=0.995, update_every=10)
        ema.load_state_dict(checkpoint['ema'])
        ema.ema_model.eval()
        print(f"Model loaded successfully with EMA on {device}")
        return ema.ema_model, config
    elif 'model' in checkpoint:
        diffusion.load_state_dict(checkpoint['model'])
        diffusion.eval()
        print(f"Model loaded successfully on {device}")
        return diffusion, config
    else:
        diffusion.load_state_dict(checkpoint)
        diffusion.eval()
        print(f"Model loaded successfully on {device}")
        return diffusion, config


def preprocess_image(image_path, image_size, original_size=None):
    """
    Preprocess a single image for model input.
    
    Args:
        image_path (str or Path): Path to the input image
        image_size (int): Target size for the image (will be resized to image_size x image_size)
        original_size (tuple): Optional original size to resize to for output, if None uses image's natural size
    
    Returns:
        tuple: (preprocessed_tensor, original_tensor, original_image) 
               - preprocessed_tensor: Tensor ready for model input (1, 3, H, W) at image_size
               - original_tensor: Tensor at original resolution for warping (1, 3, H_orig, W_orig)
               - original_image: PIL Image object of the original image
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_size = img.size
    
    if original_size is None:
        original_size = img_size
    
    # Transform for model input (downsampled)
    transform_model = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    
    # Transform for output display (original or specified resolution)
    transform_output = T.Compose([
        T.Resize(original_size) if original_size != img_size else T.Lambda(lambda x: x),
        T.ToTensor()
    ])
    
    # Process image
    img_tensor_model = transform_model(img).unsqueeze(0)  # For model: (1, 3, image_size, image_size)
    img_tensor_output = transform_output(img).unsqueeze(0)  # For warping: (1, 3, H, W)
    
    print(f"Loaded image: {image_path}")
    print(f"Original size: {img_size}, Model input size: {image_size}x{image_size}, Output size: {original_size}")
    
    return img_tensor_model, img_tensor_output, img


def run_inference(diffusion, img_tensor, device, cond_scale=1.0, rescaled_phi=0.7):
    """
    Run inference on a preprocessed image tensor to predict optical flow.
    
    Args:
        diffusion: RS_Diffusion model
        img_tensor: Preprocessed image tensor (1, 3, H, W)
        device: Device to run inference on
        cond_scale: Classifier-free guidance scale
        rescaled_phi: Rescaled phi parameter for guidance
    
    Returns:
        torch.Tensor: Predicted optical flow (1, 2, H, W)
    """
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        # Run diffusion sampling to predict flow
        # For single image inference, we use class 0
        classes = torch.zeros(1, dtype=torch.long).to(device)
        
        # Use the sample method with guidance to get flow prediction
        predicted_flow = diffusion.sample(
            classes=classes,
            condition=img_tensor,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi
        )
        
    return predicted_flow


def save_results(output_dir, image_name, corrected_img_tensor, flow_np, original_img_tensor, save_flow_viz=True):
    """
    Save inference results to disk.
    
    Args:
        output_dir (Path): Output directory
        image_name (str): Name of the input image (without extension)
        corrected_img_tensor: Corrected image tensor (1, 3, H, W)
        flow_np: Optical flow array (H, W, 2)
        original_img_tensor: Original RS image tensor (1, 3, H, W)
        save_flow_viz: Whether to save flow visualization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save corrected image
    corrected_path = output_dir / f"{image_name}_corrected.png"
    utils.save_image(corrected_img_tensor, str(corrected_path))
    print(f"Saved corrected image: {corrected_path}")
    
    # Save flow as numpy array
    flow_path = output_dir / f"{image_name}_flow.npy"
    np.save(flow_path, flow_np)
    print(f"Saved flow array: {flow_path}")
    
    # Save flow visualization
    if save_flow_viz:
        flow_viz = flow_to_image(flow_np)
        flow_viz_path = output_dir / f"{image_name}_flow_viz.png"
        cv2.imwrite(str(flow_viz_path), flow_viz)
        print(f"Saved flow visualization: {flow_viz_path}")
    
    # Create comparison image (RS | Corrected)
    comparison = torch.cat([original_img_tensor, corrected_img_tensor], dim=3)  # Concatenate along width
    comparison_path = output_dir / f"{image_name}_comparison.png"
    utils.save_image(comparison, str(comparison_path))
    print(f"Saved comparison image: {comparison_path}")


def process_single_image(image_path, diffusion, config, output_dir, device, args):
    """
    Process a single image through the inference pipeline.
    
    Args:
        image_path: Path to input image
        diffusion: Loaded diffusion model
        config: Configuration dictionary
        output_dir: Output directory path
        device: Device to run on
        args: Command line arguments
    """
    image_size = args.image_size if args.image_size else config["diffusion"]["image_size"]
    
    # Load the original image first to get its size
    original_img = Image.open(image_path).convert('RGB')
    
    # Preprocess - get both model input and output resolution tensors
    img_tensor_model, img_tensor_output, _ = preprocess_image(
        image_path, 
        image_size,
        original_size=original_img.size
    )
    
    # Move to device
    img_tensor_output = img_tensor_output.to(device)
    
    # Run inference to get flow prediction
    print(f"Running inference on {Path(image_path).name}...")
    predicted_flow = run_inference(
        diffusion, 
        img_tensor_model, 
        device,
        cond_scale=args.cond_scale,
        rescaled_phi=args.rescaled_phi
    )
    
    # Upsample flow to match output resolution
    flow_upsampled = upsample2d_flow_as(
        predicted_flow, 
        img_tensor_output, 
        mode="bilinear", 
        if_rate=True
    )
    
    # Warp the original image with the predicted flow to generate corrected image
    corrected_img_tensor = flow_warp(
        img_tensor_output, 
        flow_upsampled, 
        pad="zeros", 
        mode="bilinear"
    )
    
    # Convert flow to numpy for saving
    flow_np = flow_upsampled[0].detach().cpu().numpy().transpose(1, 2, 0)  # (2, H, W) -> (H, W, 2)
    
    # Save results
    image_name = Path(image_path).stem
    save_results(
        output_dir, 
        image_name, 
        corrected_img_tensor, 
        flow_np, 
        img_tensor_output, 
        args.save_flow_viz
    )
    
    print(f"Processing complete for {Path(image_path).name}\n")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run RS-Diffusion inference on rolling shutter images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or folder containing images"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default="config/RS_real_config_teat.yaml",
        help="Path to configuration YAML file (default: config/RS_real_config_teat.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_output",
        help="Output directory for results (default: inference_output/)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--cond_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (default: 1.0, use 1.0 for no guidance)"
    )
    parser.add_argument(
        "--rescaled_phi",
        type=float,
        default=0.7,
        help="Rescaled phi for guidance (default: 0.7)"
    )
    parser.add_argument(
        "--save_flow_viz",
        type=bool,
        default=True,
        help="Save flow visualization images (default: True)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Image size for processing (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint does not exist: {checkpoint_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config file does not exist: {config_path}")
        sys.exit(1)
    
    # Load model
    print("="*60)
    print("RS-Diffusion Inference")
    print("="*60)
    diffusion, config = load_model(args.config, args.checkpoint, args.device)
    
    # Determine input type and get image paths
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        # Support common image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_path.glob(ext))
            image_paths.extend(input_path.glob(ext.upper()))
        
        if not image_paths:
            print(f"Error: No images found in directory: {input_path}")
            sys.exit(1)
    else:
        print(f"Error: Invalid input path: {input_path}")
        sys.exit(1)
    
    print(f"\nFound {len(image_paths)} image(s) to process")
    print(f"Output directory: {args.output}")
    print("="*60)
    
    # Process images
    output_dir = Path(args.output)
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            process_single_image(image_path, diffusion, config, output_dir, args.device, args)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    print("="*60)
    print(f"Inference complete! Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
