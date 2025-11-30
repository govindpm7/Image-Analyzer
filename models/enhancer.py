"""
Low-Light Image Enhancement Model
Uses U-Net architecture for enhancement
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path


class UNet(nn.Module):
    """Simple U-Net for image enhancement"""
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder (input channels match skip connection after reduction)
        # After reducing upsampled features to match skip connections, decoder takes combined features
        self.dec4 = self.conv_block(256, 256)  # 256 (reduced) + 256 (skip) -> 256 after addition
        self.dec3 = self.conv_block(128, 128)  # 128 (reduced) + 128 (skip) -> 128 after addition
        self.dec2 = self.conv_block(64, 64)    # 64 (reduced) + 64 (skip) -> 64 after addition
        self.dec1 = nn.Conv2d(64, 3, kernel_size=1)
        
        # Channel reduction layers for skip connections (to match channel dimensions)
        self.reduce4 = nn.Conv2d(512, 256, kernel_size=1)
        self.reduce3 = nn.Conv2d(256, 128, kernel_size=1)
        self.reduce2 = nn.Conv2d(128, 64, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder with skip connections
        # Upsample e4 (512 channels) first
        up4 = self.upsample(e4)
        # Match spatial dimensions with e3
        if up4.size()[2:] != e3.size()[2:]:
            up4 = nn.functional.interpolate(up4, size=e3.size()[2:], mode='bilinear', align_corners=True)
        # Reduce channels from 512 to 256 to match e3 before addition
        up4_reduced = self.reduce4(up4)
        # Verify channel dimensions match before addition
        if up4_reduced.size(1) != e3.size(1):
            raise RuntimeError(
                f"Channel dimension mismatch in decoder: up4_reduced has {up4_reduced.size(1)} channels, "
                f"but e3 has {e3.size(1)} channels. This indicates a model architecture issue. "
                f"Expected reduce4 to output 256 channels but got {up4_reduced.size(1)}."
            )
        d4 = self.dec4(up4_reduced + e3)
        
        # Upsample d4 (256 channels)
        up3 = self.upsample(d4)
        # Match spatial dimensions with e2
        if up3.size()[2:] != e2.size()[2:]:
            up3 = nn.functional.interpolate(up3, size=e2.size()[2:], mode='bilinear', align_corners=True)
        # Reduce channels from 256 to 128 to match e2 before addition
        up3_reduced = self.reduce3(up3)
        # Verify channel dimensions match before addition
        if up3_reduced.size(1) != e2.size(1):
            raise RuntimeError(
                f"Channel dimension mismatch in decoder: up3_reduced has {up3_reduced.size(1)} channels, "
                f"but e2 has {e2.size(1)} channels. This indicates a model architecture issue. "
                f"Expected reduce3 to output 128 channels but got {up3_reduced.size(1)}."
            )
        d3 = self.dec3(up3_reduced + e2)
        
        # Upsample d3 (128 channels)
        up2 = self.upsample(d3)
        # Match spatial dimensions with e1
        if up2.size()[2:] != e1.size()[2:]:
            up2 = nn.functional.interpolate(up2, size=e1.size()[2:], mode='bilinear', align_corners=True)
        # Reduce channels from 128 to 64 to match e1 before addition
        up2_reduced = self.reduce2(up2)
        # Verify channel dimensions match before addition
        if up2_reduced.size(1) != e1.size(1):
            raise RuntimeError(
                f"Channel dimension mismatch in decoder: up2_reduced has {up2_reduced.size(1)} channels, "
                f"but e1 has {e1.size(1)} channels. This indicates a model architecture issue. "
                f"Expected reduce2 to output 64 channels but got {up2_reduced.size(1)}."
            )
        d2 = self.dec2(up2_reduced + e1)
        
        d1 = self.dec1(d2)
        
        return torch.sigmoid(d1)


class LowLightEnhancer:
    def __init__(self, device=None, architecture='unet'):
        """
        Initialize low-light enhancer
        
        Args:
            device: torch device (cuda/cpu)
            architecture: 'unet' or 'curve' - which model architecture to use
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.architecture = architecture.lower()
        
        # Initialize model based on architecture choice
        if self.architecture == 'curve':
            self.model = EnhancementCurveNet().to(self.device)
        else:
            self.model = UNet().to(self.device)
            
        self.model.eval()
        self.weights_loaded = False  # Track if weights were successfully loaded
        self.last_method_used = "CV"  # Track which method was used
        
    def load_weights(self, weights_path):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            
            # Check if checkpoint specifies architecture
            if isinstance(checkpoint, dict):
                if 'architecture' in checkpoint:
                    saved_arch = checkpoint['architecture']
                    if saved_arch != self.architecture:
                        print(f"⚠ Warning: Checkpoint architecture ({saved_arch}) doesn't match current ({self.architecture})")
                        print(f"   Reinitializing model with {saved_arch} architecture...")
                        # Reinitialize with correct architecture
                        if saved_arch == 'curve':
                            self.model = EnhancementCurveNet().to(self.device)
                        else:
                            self.model = UNet().to(self.device)
                        self.architecture = saved_arch
                        self.model.eval()
                
                # Handle checkpoint dict format
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                # Direct state_dict
                self.model.load_state_dict(checkpoint)
                
            self.weights_loaded = True  # Mark as successfully loaded
            print(f"✓ Loaded enhancer weights from {weights_path} (architecture: {self.architecture})")
        except FileNotFoundError:
            print(f"⚠ Warning: Weights not found at {weights_path}. Using classical enhancement.")
            self.weights_loaded = False
        except Exception as e:
            print(f"⚠ Warning: Could not load weights: {e}. Using classical enhancement.")
            self.weights_loaded = False
    
    def enhance(self, image, preserve_colors=True, max_brightness=220, use_hybrid=False, ml_strength=0.05):
        """
        Enhance low-light image using CV preprocessing + optional ML refinement
        
        Args:
            image: numpy array (BGR format from OpenCV)
            preserve_colors: If True, uses more conservative enhancement to avoid over-saturation
            max_brightness: Maximum average brightness value (0-255) to prevent over-enhancement
            use_hybrid: If True, uses CV preprocessing + ML refinement (recommended)
            ml_strength: Strength of ML adjustment (0.0-1.0), lower = more conservative
            
        Returns:
            numpy array: Enhanced image (BGR format)
        """
        # Always start with CV preprocessing for stable base enhancement
        cv_enhanced = self._classical_enhancement(image, preserve_colors=preserve_colors)
        method_used = "CV"
        
        # Use ML model for minor refinements if weights are loaded and hybrid mode is enabled
        if self.weights_loaded and use_hybrid:
            try:
                # Use CV result as input to ML for refinement
                original_shape = image.shape[:2]
                
                # Use higher resolution to prevent pixelation (512x512 instead of 256x256)
                # Calculate target size maintaining aspect ratio, max 512px
                max_dim = max(original_shape)
                if max_dim > 512:
                    scale = 512 / max_dim
                    target_size = (int(original_shape[1] * scale), int(original_shape[0] * scale))
                else:
                    target_size = (original_shape[1], original_shape[0])
                
                # Resize CV enhanced image to target size
                cv_resized = cv2.resize(cv_enhanced, target_size, interpolation=cv2.INTER_LINEAR)
                cv_rgb = cv2.cvtColor(cv_resized, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1]
                cv_tensor = torch.from_numpy(cv_rgb.astype(np.float32) / 255.0)
                cv_tensor = cv_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # Get ML refinement (treat as residual adjustment)
                with torch.no_grad():
                    ml_output = self.model(cv_tensor)
                    ml_output = ml_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    ml_output = (ml_output * 255.0).astype(np.float32)
                
                # Convert CV result to float for blending
                cv_float = cv_rgb.astype(np.float32)
                
                # Blend: CV (base) + ML (refinement) with very low strength
                # Very low ml_strength (0.05-0.1) = ML makes tiny adjustments, preserving CV quality
                blended = cv_float * (1.0 - ml_strength) + ml_output * ml_strength
                blended = np.clip(blended, 0, 255).astype(np.uint8)
                
                # Convert RGB to BGR and resize back to original using high-quality interpolation
                blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
                enhanced = cv2.resize(blended_bgr, (original_shape[1], original_shape[0]), 
                                     interpolation=cv2.INTER_LANCZOS4)  # High-quality upscaling
                method_used = "CV+ML"
            except Exception as e:
                print(f"Error in ML refinement: {e}")
                enhanced = cv_enhanced
                method_used = "CV"
        else:
            enhanced = cv_enhanced
            method_used = "CV"
        
        # Apply brightness limiting to prevent over-enhancement
        enhanced = self._limit_brightness(enhanced, max_brightness=max_brightness)
        
        # Store method used for display
        self.last_method_used = method_used
        
        return enhanced
    
    def _classical_enhancement(self, image, preserve_colors=True):
        """Fallback: Classical CV enhancement using CLAHE and gamma correction"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Use more conservative CLAHE settings to avoid over-saturation
        clip_limit = 2.0 if preserve_colors else 3.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # More conservative gamma correction
        gamma = 1.3 if preserve_colors else 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        # Reduce saturation if preserve_colors is True
        if preserve_colors:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            # Reduce saturation by 10% to keep it more natural
            s = cv2.multiply(s, 0.9)
            enhanced = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        
        return enhanced
    
    def _limit_brightness(self, image, max_brightness=220):
        """
        Limit the overall brightness of an image to prevent over-enhancement
        
        Args:
            image: numpy array (BGR format)
            max_brightness: Maximum average brightness value (0-255)
            
        Returns:
            numpy array: Brightness-limited image
        """
        # Calculate current average brightness in HSV space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        current_brightness = np.mean(hsv[:,:,2])
        
        # If brightness exceeds limit, tone it down
        if current_brightness > max_brightness:
            # Calculate scaling factor to bring brightness down to max_brightness
            # Use a smooth scaling to avoid harsh transitions
            scale_factor = max_brightness / current_brightness
            
            # Apply scaling to V channel (brightness)
            v = hsv[:,:,2].astype(np.float32)
            v = v * scale_factor
            v = np.clip(v, 0, 255).astype(np.uint8)
            
            # Reconstruct HSV and convert back to BGR
            hsv[:,:,2] = v
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image


