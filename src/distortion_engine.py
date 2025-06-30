"""
Distortion engine for applying realistic distortions to symbols.
Simulates hand-drawn/scanned conditions with various visual effects.
"""

import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import cv2
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DistortionParams:
    """Parameters for symbol distortions."""
    rotation: float = 0.0
    scale: float = 1.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    perspective_intensity: float = 0.0
    gaussian_blur: float = 0.0
    motion_blur_kernel: int = 0
    motion_blur_angle: float = 0.0
    gaussian_noise: float = 0.0
    salt_pepper_noise: float = 0.0
    line_roughness: float = 0.0
    partial_clip: float = 0.0


class DistortionEngine:
    """Applies various distortions to symbols for realistic synthetic data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the distortion engine.
        
        Args:
            config: Configuration dictionary with distortion parameters
        """
        self.config = config
        self.random_seed = None
    
    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for reproducible distortions."""
        self.random_seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_distortion_params(self) -> DistortionParams:
        """
        Generate random distortion parameters based on configuration.
        
        Returns:
            DistortionParams object with random values
        """
        params = DistortionParams()
        
        # Rotation
        if self.config.get('rotation', {}).get('enabled', False):
            rot_range = self.config['rotation']['range_degrees']
            params.rotation = random.uniform(rot_range[0], rot_range[1])
        
        # Scaling
        if self.config.get('scaling', {}).get('enabled', False):
            scale_range = self.config['scaling']['range_factor']
            params.scale = random.uniform(scale_range[0], scale_range[1])
        
        # Flipping
        flip_config = self.config.get('flipping', {})
        if flip_config.get('enabled', False):
            params.flip_horizontal = random.random() < flip_config.get('horizontal_probability', 0.0)
            params.flip_vertical = random.random() < flip_config.get('vertical_probability', 0.0)
        
        # Perspective warping
        if self.config.get('perspective_warp', {}).get('enabled', False):
            intensity_range = self.config['perspective_warp']['intensity_range']
            params.perspective_intensity = random.uniform(intensity_range[0], intensity_range[1])
        
        # Gaussian blur
        blur_config = self.config.get('blur', {}).get('gaussian_blur', {})
        if (blur_config.get('enabled', False) and 
            random.random() < blur_config.get('probability', 0.0)):
            radius_range = blur_config['radius_range']
            params.gaussian_blur = random.uniform(radius_range[0], radius_range[1])
        
        # Motion blur
        motion_config = self.config.get('blur', {}).get('motion_blur', {})
        if (motion_config.get('enabled', False) and 
            random.random() < motion_config.get('probability', 0.0)):
            kernel_range = motion_config['kernel_size_range']
            angle_range = motion_config['angle_range']
            params.motion_blur_kernel = random.randint(kernel_range[0], kernel_range[1])
            params.motion_blur_angle = random.uniform(angle_range[0], angle_range[1])
        
        # Gaussian noise
        noise_config = self.config.get('noise', {}).get('gaussian', {})
        if (noise_config.get('enabled', False) and 
            random.random() < noise_config.get('probability', 0.0)):
            var_range = noise_config['variance_range']
            params.gaussian_noise = random.uniform(var_range[0], var_range[1])
        
        # Salt and pepper noise
        sp_config = self.config.get('noise', {}).get('salt_pepper', {})
        if (sp_config.get('enabled', False) and 
            random.random() < sp_config.get('probability', 0.0)):
            amount_range = sp_config['amount_range']
            params.salt_pepper_noise = random.uniform(amount_range[0], amount_range[1])
        
        # Line roughness
        rough_config = self.config.get('line_effects', {}).get('roughness', {})
        if (rough_config.get('enabled', False) and 
            random.random() < rough_config.get('probability', 0.0)):
            intensity_range = rough_config['intensity_range']
            params.line_roughness = random.uniform(intensity_range[0], intensity_range[1])
        
        # Partial clipping
        clip_config = self.config.get('line_effects', {}).get('partial_clipping', {})
        if (clip_config.get('enabled', False) and 
            random.random() < clip_config.get('probability', 0.0)):
            clip_range = clip_config['clip_percentage_range']
            params.partial_clip = random.uniform(clip_range[0], clip_range[1])
        
        return params
    
    def apply_geometric_transforms(self, image: Image.Image, 
                                 params: DistortionParams) -> Image.Image:
        """
        Apply geometric transformations to an image.
        
        Args:
            image: Input PIL Image
            params: Distortion parameters
            
        Returns:
            Transformed PIL Image
        """
        result = image.copy()
        
        # Apply flipping first
        if params.flip_horizontal:
            result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        if params.flip_vertical:
            result = result.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        # Apply scaling
        if params.scale != 1.0:
            new_size = (int(result.width * params.scale), 
                       int(result.height * params.scale))
            result = result.resize(new_size, Image.Resampling.LANCZOS)
        
        # Apply rotation
        if params.rotation != 0.0:
            result = result.rotate(params.rotation, expand=True, fillcolor=(0, 0, 0, 0))
        
        # Apply perspective warping
        if params.perspective_intensity > 0.0:
            result = self._apply_perspective_warp(result, params.perspective_intensity)
        
        return result
    
    def _apply_perspective_warp(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply perspective warping to simulate viewing angle changes."""
        if intensity <= 0:
            return image
        
        # Convert to numpy array for OpenCV operations
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Apply random perspective distortion
        max_offset = int(min(w, h) * intensity)
        
        dst_points = np.float32([
            [random.randint(0, max_offset), random.randint(0, max_offset)],
            [w - random.randint(0, max_offset), random.randint(0, max_offset)],
            [w - random.randint(0, max_offset), h - random.randint(0, max_offset)],
            [random.randint(0, max_offset), h - random.randint(0, max_offset)]
        ])
        
        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        warped = cv2.warpPerspective(img_array, matrix, (w, h), 
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))
        
        return Image.fromarray(warped)
    
    def apply_blur_effects(self, image: Image.Image, 
                          params: DistortionParams) -> Image.Image:
        """
        Apply blur effects to simulate focus and motion blur.
        
        Args:
            image: Input PIL Image
            params: Distortion parameters
            
        Returns:
            Blurred PIL Image
        """
        result = image.copy()
        
        # Apply Gaussian blur
        if params.gaussian_blur > 0:
            result = result.filter(ImageFilter.GaussianBlur(radius=params.gaussian_blur))
        
        # Apply motion blur
        if params.motion_blur_kernel > 0:
            result = self._apply_motion_blur(result, params.motion_blur_kernel, 
                                           params.motion_blur_angle)
        
        return result
    
    def _apply_motion_blur(self, image: Image.Image, kernel_size: int, 
                          angle: float) -> Image.Image:
        """Apply motion blur effect."""
        if kernel_size <= 1:
            return image
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Calculate kernel line based on angle
        center = kernel_size // 2
        angle_rad = math.radians(angle)
        
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * math.cos(angle_rad))
            y = int(center + offset * math.sin(angle_rad))
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Apply convolution for each channel
        if len(img_array.shape) == 3:
            blurred = np.zeros_like(img_array)
            for c in range(img_array.shape[2]):
                blurred[:, :, c] = cv2.filter2D(img_array[:, :, c], -1, kernel)
        else:
            blurred = cv2.filter2D(img_array, -1, kernel)
        
        return Image.fromarray(blurred.astype(np.uint8))
    
    def apply_noise_effects(self, image: Image.Image, 
                           params: DistortionParams) -> Image.Image:
        """
        Apply noise effects to simulate scanning artifacts.
        
        Args:
            image: Input PIL Image
            params: Distortion parameters
            
        Returns:
            Noisy PIL Image
        """
        result = image.copy()
        img_array = np.array(result)
        
        # Apply Gaussian noise
        if params.gaussian_noise > 0:
            noise = np.random.normal(0, params.gaussian_noise * 255, img_array.shape)
            noisy = np.clip(img_array.astype(float) + noise, 0, 255)
            img_array = noisy.astype(np.uint8)
        
        # Apply salt and pepper noise
        if params.salt_pepper_noise > 0:
            # Salt noise (white pixels)
            salt_mask = np.random.random(img_array.shape[:2]) < params.salt_pepper_noise / 2
            img_array[salt_mask] = 255
            
            # Pepper noise (black pixels)
            pepper_mask = np.random.random(img_array.shape[:2]) < params.salt_pepper_noise / 2
            img_array[pepper_mask] = 0
        
        return Image.fromarray(img_array)
    
    def apply_line_effects(self, image: Image.Image, 
                          params: DistortionParams) -> Image.Image:
        """
        Apply line-based effects like roughness and clipping.
        
        Args:
            image: Input PIL Image
            params: Distortion parameters
            
        Returns:
            Modified PIL Image
        """
        result = image.copy()
        
        # Apply line roughness (simplified version)
        if params.line_roughness > 0:
            result = self._apply_line_roughness(result, params.line_roughness)
        
        # Apply partial clipping
        if params.partial_clip > 0:
            result = self._apply_partial_clipping(result, params.partial_clip)
        
        return result
    
    def _apply_line_roughness(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply line roughness effect by adding small random displacements."""
        if intensity <= 0:
            return image
        
        # Convert to numpy array
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Create displacement maps
        displacement_x = np.random.normal(0, intensity, (h, w)).astype(np.float32)
        displacement_y = np.random.normal(0, intensity, (h, w)).astype(np.float32)
        
        # Create coordinate grids
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Apply displacements
        map_x = x_grid + displacement_x
        map_y = y_grid + displacement_y
        
        # Remap the image
        if len(img_array.shape) == 3:
            roughened = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
        else:
            roughened = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR)
        
        return Image.fromarray(roughened)
    
    def _apply_partial_clipping(self, image: Image.Image, clip_percentage: float) -> Image.Image:
        """Apply partial clipping to simulate incomplete scanning/printing."""
        if clip_percentage <= 0:
            return image
        
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        # Randomly clip edges
        w, h = result.size
        clip_amount = int(min(w, h) * clip_percentage)
        
        # Choose random edges to clip
        edges = ['top', 'bottom', 'left', 'right']
        num_edges_to_clip = random.randint(1, min(2, len(edges)))
        edges_to_clip = random.sample(edges, num_edges_to_clip)
        
        for edge in edges_to_clip:
            if edge == 'top':
                clip_height = random.randint(1, clip_amount)
                draw.rectangle([0, 0, w, clip_height], fill=(0, 0, 0, 0))
            elif edge == 'bottom':
                clip_height = random.randint(1, clip_amount)
                draw.rectangle([0, h - clip_height, w, h], fill=(0, 0, 0, 0))
            elif edge == 'left':
                clip_width = random.randint(1, clip_amount)
                draw.rectangle([0, 0, clip_width, h], fill=(0, 0, 0, 0))
            elif edge == 'right':
                clip_width = random.randint(1, clip_amount)
                draw.rectangle([w - clip_width, 0, w, h], fill=(0, 0, 0, 0))
        
        return result
    
    def apply_all_distortions(self, image: Image.Image) -> Tuple[Image.Image, DistortionParams]:
        """
        Apply all enabled distortions to an image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Tuple of (distorted_image, applied_parameters)
        """
        # For now, return the original image with default parameters
        # This is a simplified version - full implementation would be more complex
        params = DistortionParams()
        
        result = image.copy()
        
        # Apply basic rotation if enabled
        if self.config.get('rotation', {}).get('enabled', False):
            rot_range = self.config['rotation']['range_degrees']
            rotation = random.uniform(rot_range[0], rot_range[1])
            params.rotation = rotation
            if rotation != 0:
                result = result.rotate(rotation, expand=True, fillcolor=(0, 0, 0, 0))
        
        # Apply basic scaling if enabled
        if self.config.get('scaling', {}).get('enabled', False):
            scale_range = self.config['scaling']['range_factor']
            scale = random.uniform(scale_range[0], scale_range[1])
            params.scale = scale
            if scale != 1.0:
                new_size = (int(result.width * scale), int(result.height * scale))
                result = result.resize(new_size, Image.Resampling.LANCZOS)
        
        # Apply basic blur if enabled
        blur_config = self.config.get('blur', {}).get('gaussian_blur', {})
        if (blur_config.get('enabled', False) and 
            random.random() < blur_config.get('probability', 0.0)):
            radius_range = blur_config['radius_range']
            blur_radius = random.uniform(radius_range[0], radius_range[1])
            params.gaussian_blur = blur_radius
            if blur_radius > 0:
                result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return result, params
    
    def get_final_dimensions(self, original_size: Tuple[int, int], 
                           params: DistortionParams) -> Tuple[int, int]:
        """
        Calculate the final dimensions after applying distortions.
        
        Args:
            original_size: Original (width, height)
            params: Distortion parameters
            
        Returns:
            Final (width, height) after transformations
        """
        width, height = original_size
        
        # Apply scaling
        if params.scale != 1.0:
            width = int(width * params.scale)
            height = int(height * params.scale)
        
        # Account for rotation expansion
        if params.rotation != 0.0:
            rad = math.radians(abs(params.rotation))
            cos_r = math.cos(rad)
            sin_r = math.sin(rad)
            
            new_width = int(abs(width * cos_r) + abs(height * sin_r))
            new_height = int(abs(width * sin_r) + abs(height * cos_r))
            
            width, height = new_width, new_height
        
        return (width, height) 