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
        
        # Decoder
        self.dec4 = self.conv_block(512, 256)
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = nn.Conv2d(64, 3, kernel_size=1)
        
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
        
        # Decoder with skip connections (crop to match dimensions)
        up4 = self.upsample(e4)
        # Crop upsampled tensor to match e3 dimensions
        _, _, h3, w3 = e3.size()
        _, _, h_up, w_up = up4.size()
        up4_cropped = up4[:, :, :h3, :w3] if h_up >= h3 and w_up >= w3 else up4
        if up4_cropped.size() != e3.size():
            up4_cropped = nn.functional.interpolate(up4_cropped, size=(h3, w3), mode='bilinear', align_corners=True)
        d4 = self.dec4(up4_cropped + e3)
        
        up3 = self.upsample(d4)
        _, _, h2, w2 = e2.size()
        _, _, h_up, w_up = up3.size()
        up3_cropped = up3[:, :, :h2, :w2] if h_up >= h2 and w_up >= w2 else up3
        if up3_cropped.size() != e2.size():
            up3_cropped = nn.functional.interpolate(up3_cropped, size=(h2, w2), mode='bilinear', align_corners=True)
        d3 = self.dec3(up3_cropped + e2)
        
        up2 = self.upsample(d3)
        _, _, h1, w1 = e1.size()
        _, _, h_up, w_up = up2.size()
        up2_cropped = up2[:, :, :h1, :w1] if h_up >= h1 and w_up >= w1 else up2
        if up2_cropped.size() != e1.size():
            up2_cropped = nn.functional.interpolate(up2_cropped, size=(h1, w1), mode='bilinear', align_corners=True)
        d2 = self.dec2(up2_cropped + e1)
        
        d1 = self.dec1(d2)
        
        return torch.sigmoid(d1)


class LowLightEnhancer:
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        self.model.eval()
        
    def load_weights(self, weights_path):
        """Load trained model weights"""
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"✓ Loaded enhancer weights from {weights_path}")
        except FileNotFoundError:
            print(f"⚠ Warning: Weights not found at {weights_path}. Using classical enhancement.")
        except Exception as e:
            print(f"⚠ Warning: Could not load weights: {e}. Using classical enhancement.")
    
    def enhance(self, image, preserve_colors=True, max_brightness=220):
        """
        Enhance low-light image
        
        Args:
            image: numpy array (BGR format from OpenCV)
            preserve_colors: If True, uses more conservative enhancement to avoid over-saturation
            max_brightness: Maximum average brightness value (0-255) to prevent over-enhancement
            
        Returns:
            numpy array: Enhanced image (BGR format)
        """
        # Since model isn't trained, always use classical enhancement
        # Check if weights exist - if not, use classical method
        weights_path = Path('weights/enhancer_best.pth')
        if not weights_path.exists():
            enhanced = self._classical_enhancement(image, preserve_colors=preserve_colors)
        else:
            try:
                # Preprocess
                original_shape = image.shape[:2]
                img_resized = cv2.resize(image, (256, 256))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1]
                img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # Enhance
                with torch.no_grad():
                    enhanced = self.model(img_tensor)
                    enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    enhanced = (enhanced * 255.0).astype(np.uint8)
                
                # Convert RGB to BGR and resize back
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
                enhanced_bgr = cv2.resize(enhanced_bgr, (original_shape[1], original_shape[0]))
                enhanced = enhanced_bgr
            except Exception as e:
                print(f"Error in enhancement: {e}")
                enhanced = self._classical_enhancement(image, preserve_colors=preserve_colors)
        
        # Apply brightness limiting to prevent over-enhancement
        enhanced = self._limit_brightness(enhanced, max_brightness=max_brightness)
        
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


