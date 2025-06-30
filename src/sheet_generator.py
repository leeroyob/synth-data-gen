"""
Sheet generator for creating blank engineering sheets.
Handles creation of base sheets and loading of background images.
"""

from PIL import Image, ImageDraw
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import random


class SheetGenerator:
    """Generates blank engineering sheets for symbol placement."""
    
    def __init__(self, width_px: int, height_px: int, dpi: int = 300):
        """
        Initialize the sheet generator.
        
        Args:
            width_px: Sheet width in pixels
            height_px: Sheet height in pixels
            dpi: Dots per inch for the sheet
        """
        self.width_px = width_px
        self.height_px = height_px
        self.dpi = dpi
        self.backgrounds_dir = None
        self.available_backgrounds = []
    
    def set_backgrounds_directory(self, backgrounds_dir: str) -> None:
        """
        Set the directory containing background images.
        
        Args:
            backgrounds_dir: Path to the backgrounds directory
        """
        self.backgrounds_dir = Path(backgrounds_dir)
        if self.backgrounds_dir.exists():
            # Find all image files in the backgrounds directory
            image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
            self.available_backgrounds = [
                f for f in self.backgrounds_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ]
        else:
            self.available_backgrounds = []
    
    def create_blank_sheet(self, background_color: str = "white") -> Image.Image:
        """
        Create a blank engineering sheet.
        
        Args:
            background_color: Background color for the sheet
            
        Returns:
            PIL Image object representing the blank sheet
        """
        # Create a new image with the specified dimensions
        sheet = Image.new('RGB', (self.width_px, self.height_px), background_color)
        
        # Optionally add subtle grid lines or margins for debugging
        # This can be enabled via configuration later
        
        return sheet
    
    def create_sheet_with_background(self, background_path: Optional[str] = None) -> Image.Image:
        """
        Create a sheet with a background image.
        
        Args:
            background_path: Path to specific background image. If None, chooses randomly.
            
        Returns:
            PIL Image object with background
        """
        if background_path is None:
            if not self.available_backgrounds:
                # No backgrounds available, return blank sheet
                return self.create_blank_sheet()
            
            # Choose a random background
            bg_path = random.choice(self.available_backgrounds)
        else:
            bg_path = Path(background_path)
        
        try:
            # Load the background image
            background = Image.open(bg_path)
            
            # Convert to RGB if necessary
            if background.mode != 'RGB':
                background = background.convert('RGB')
            
            # Resize background to match sheet dimensions
            if background.size != (self.width_px, self.height_px):
                background = background.resize(
                    (self.width_px, self.height_px),
                    Image.Resampling.LANCZOS
                )
            
            return background
            
        except Exception as e:
            print(f"Warning: Could not load background {bg_path}: {e}")
            return self.create_blank_sheet()
    
    def add_debug_grid(self, sheet: Image.Image, grid_spacing_inches: float = 1.0) -> Image.Image:
        """
        Add a subtle grid to the sheet for debugging purposes.
        
        Args:
            sheet: The sheet image to add grid to
            grid_spacing_inches: Spacing between grid lines in inches
            
        Returns:
            Sheet with grid overlay
        """
        # Create a copy to avoid modifying the original
        sheet_with_grid = sheet.copy()
        draw = ImageDraw.Draw(sheet_with_grid)
        
        # Calculate grid spacing in pixels
        grid_spacing_px = int(grid_spacing_inches * self.dpi)
        
        # Grid color (very light gray)
        grid_color = (240, 240, 240)
        
        # Draw vertical lines
        for x in range(0, self.width_px, grid_spacing_px):
            draw.line([(x, 0), (x, self.height_px)], fill=grid_color, width=1)
        
        # Draw horizontal lines
        for y in range(0, self.height_px, grid_spacing_px):
            draw.line([(0, y), (self.width_px, y)], fill=grid_color, width=1)
        
        return sheet_with_grid
    
    def add_margins_outline(self, sheet: Image.Image, margin_inches: float) -> Image.Image:
        """
        Add margin outlines to the sheet for debugging.
        
        Args:
            sheet: The sheet image to add margins to
            margin_inches: Margin size in inches
            
        Returns:
            Sheet with margin outlines
        """
        sheet_with_margins = sheet.copy()
        draw = ImageDraw.Draw(sheet_with_margins)
        
        # Calculate margin in pixels
        margin_px = int(margin_inches * self.dpi)
        
        # Margin outline color (light blue)
        margin_color = (200, 200, 255)
        
        # Draw margin rectangle
        draw.rectangle([
            (margin_px, margin_px),
            (self.width_px - margin_px, self.height_px - margin_px)
        ], outline=margin_color, width=2)
        
        # Draw center line (bottom half boundary)
        center_y = self.height_px // 2
        draw.line([
            (margin_px, center_y),
            (self.width_px - margin_px, center_y)
        ], fill=margin_color, width=2)
        
        return sheet_with_margins
    
    def get_sheet_info(self) -> dict:
        """
        Get information about the sheet dimensions and settings.
        
        Returns:
            Dictionary with sheet information
        """
        return {
            'width_px': self.width_px,
            'height_px': self.height_px,
            'dpi': self.dpi,
            'width_inches': self.width_px / self.dpi,
            'height_inches': self.height_px / self.dpi,
            'available_backgrounds': len(self.available_backgrounds)
        }
    
    def generate_sheet(self, 
                      use_background: bool = False,
                      background_path: Optional[str] = None,
                      add_debug_elements: bool = False,
                      margin_inches: float = 0.75) -> Image.Image:
        """
        Generate a complete sheet with optional background and debug elements.
        
        Args:
            use_background: Whether to use a background image
            background_path: Specific background path (if None, chooses randomly)
            add_debug_elements: Whether to add grid and margin outlines
            margin_inches: Margin size for debug outlines
            
        Returns:
            Generated sheet image
        """
        if use_background:
            sheet = self.create_sheet_with_background(background_path)
        else:
            sheet = self.create_blank_sheet()
        
        if add_debug_elements:
            sheet = self.add_debug_grid(sheet)
            sheet = self.add_margins_outline(sheet, margin_inches)
        
        return sheet 