"""
Symbol manager for loading and managing engineering symbols.
Handles symbol loading, cataloging, and metadata extraction.
"""

import json
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import numpy as np


class Symbol:
    """Represents a single engineering symbol."""
    
    def __init__(self, name: str, image_path: Path, metadata: Optional[Dict] = None):
        """
        Initialize a symbol.
        
        Args:
            name: Symbol name/identifier
            image_path: Path to the symbol image file
            metadata: Optional metadata dictionary
        """
        self.name = name
        self.image_path = image_path
        self.metadata = metadata or {}
        self._image = None
        self._size = None
    
    def load_image(self) -> Image.Image:
        """
        Load the symbol image.
        
        Returns:
            PIL Image object
        """
        if self._image is None:
            self._image = Image.open(self.image_path)
            
            # Convert to RGBA for transparency support
            if self._image.mode != 'RGBA':
                self._image = self._image.convert('RGBA')
            
            # Remove non-black background if the image doesn't already have transparency
            self._image = self._remove_non_black_background(self._image)
        
        return self._image
    
    def _remove_non_black_background(self, image: Image.Image) -> Image.Image:
        """
        Remove non-black background from the image, making it transparent.
        Anything that is not black (or very close to black) will be made transparent.
        
        Args:
            image: Input PIL Image in RGBA mode
            
        Returns:
            PIL Image with non-black background removed
        """
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Create a mask for pixels that are NOT black (or close to black)
        # Consider pixels with RGB values > threshold as background
        threshold = 30  # Adjust this value to be more or less strict
        
        # Check if each pixel is close to black
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        is_black = (r <= threshold) & (g <= threshold) & (b <= threshold)
        
        # Create new alpha channel: 255 for black pixels, 0 for non-black
        new_alpha = np.where(is_black, 255, 0).astype(np.uint8)
        
        # Update the alpha channel
        img_array[:, :, 3] = new_alpha
        
        return Image.fromarray(img_array, 'RGBA')
    
    def get_size(self) -> Tuple[int, int]:
        """
        Get symbol dimensions.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        if self._size is None:
            img = self.load_image()
            self._size = img.size
        return self._size
    
    def get_class_name(self) -> str:
        """
        Extract class name from symbol name or filename.
        
        Returns:
            Class name for ML training
        """
        # Use the symbol name as class name, with some cleanup
        class_name = self.name.lower()
        
        # Remove common suffixes and prefixes
        class_name = class_name.replace('_3', '').replace('_2', '').replace('_1', '')
        
        # Replace underscores with hyphens for consistency
        class_name = class_name.replace('_', '-')
        
        return class_name
    
    def __str__(self) -> str:
        return f"Symbol({self.name}, {self.image_path.name})"


class SymbolManager:
    """Manages a collection of engineering symbols."""
    
    def __init__(self, symbols_dir: str):
        """
        Initialize the symbol manager.
        
        Args:
            symbols_dir: Path to the directory containing symbols
        """
        self.symbols_dir = Path(symbols_dir)
        self.symbols: Dict[str, Symbol] = {}
        self.manifest_data = None
        self.classes: Dict[str, List[Symbol]] = {}
    
    def load_symbols(self) -> None:
        """Load all symbols from the symbols directory."""
        if not self.symbols_dir.exists():
            raise FileNotFoundError(f"Symbols directory not found: {self.symbols_dir}")
        
        # Try to load manifest first
        self._load_manifest()
        
        # Load symbols from images directory
        images_dir = self.symbols_dir / "images"
        if images_dir.exists():
            self._load_from_images_dir(images_dir)
        else:
            # Fallback: load directly from symbols directory
            self._load_from_directory(self.symbols_dir)
        
        # Organize symbols by class
        self._organize_by_class()
        
        print(f"Loaded {len(self.symbols)} symbols in {len(self.classes)} classes")
    
    def _load_manifest(self) -> None:
        """Load the manifest.json file if it exists."""
        manifest_path = self.symbols_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    self.manifest_data = json.load(f)
                print(f"Loaded manifest with {self.manifest_data.get('total_blocks', 0)} blocks")
            except Exception as e:
                print(f"Warning: Could not load manifest: {e}")
                self.manifest_data = None
    
    def _load_from_images_dir(self, images_dir: Path) -> None:
        """Load symbols from the images directory."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        
        for image_file in images_dir.iterdir():
            if image_file.suffix.lower() in image_extensions:
                # Extract symbol name from filename
                symbol_name = image_file.stem
                
                # Try to find corresponding metadata
                metadata = self._get_symbol_metadata(symbol_name)
                
                # Create symbol object
                symbol = Symbol(symbol_name, image_file, metadata)
                self.symbols[symbol_name] = symbol
    
    def _load_from_directory(self, directory: Path) -> None:
        """Load symbols directly from a directory (fallback method)."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        
        for image_file in directory.iterdir():
            if image_file.suffix.lower() in image_extensions:
                symbol_name = image_file.stem
                symbol = Symbol(symbol_name, image_file)
                self.symbols[symbol_name] = symbol
    
    def _get_symbol_metadata(self, symbol_name: str) -> Optional[Dict]:
        """Get metadata for a specific symbol."""
        if not self.manifest_data:
            return None
        
        # Search for the symbol in the manifest
        for block in self.manifest_data.get('blocks', []):
            if block.get('name') == symbol_name:
                # Try to load additional metadata from metadata file
                metadata_file = self.symbols_dir / block.get('metadata_file', '')
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            detailed_metadata = json.load(f)
                        return {**block, **detailed_metadata}
                    except Exception:
                        pass
                return block
        
        return None
    
    def _organize_by_class(self) -> None:
        """Organize symbols by their class names."""
        self.classes.clear()
        
        for symbol in self.symbols.values():
            class_name = symbol.get_class_name()
            if class_name not in self.classes:
                self.classes[class_name] = []
            self.classes[class_name].append(symbol)
    
    def get_symbol(self, name: str) -> Optional[Symbol]:
        """
        Get a symbol by name.
        
        Args:
            name: Symbol name
            
        Returns:
            Symbol object or None if not found
        """
        return self.symbols.get(name)
    
    def get_symbols_by_class(self, class_name: str) -> List[Symbol]:
        """
        Get all symbols belonging to a specific class.
        
        Args:
            class_name: Class name
            
        Returns:
            List of symbols in the class
        """
        return self.classes.get(class_name, [])
    
    def get_all_symbols(self) -> List[Symbol]:
        """
        Get all loaded symbols.
        
        Returns:
            List of all symbols
        """
        return list(self.symbols.values())
    
    def get_all_classes(self) -> List[str]:
        """
        Get all available class names.
        
        Returns:
            List of class names
        """
        return list(self.classes.keys())
    
    def get_random_symbol(self) -> Optional[Symbol]:
        """
        Get a random symbol.
        
        Returns:
            Random symbol or None if no symbols loaded
        """
        if not self.symbols:
            return None
        
        import random
        return random.choice(list(self.symbols.values()))
    
    def get_symbol_info(self) -> Dict:
        """
        Get information about the loaded symbols.
        
        Returns:
            Dictionary with symbol statistics
        """
        total_symbols = len(self.symbols)
        total_classes = len(self.classes)
        
        class_distribution = {
            class_name: len(symbols) 
            for class_name, symbols in self.classes.items()
        }
        
        return {
            'total_symbols': total_symbols,
            'total_classes': total_classes,
            'class_distribution': class_distribution,
            'symbols_dir': str(self.symbols_dir),
            'has_manifest': self.manifest_data is not None
        }
    
    def validate_symbols(self) -> List[str]:
        """
        Validate all loaded symbols and return any issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        for name, symbol in self.symbols.items():
            try:
                # Try to load the image
                img = symbol.load_image()
                
                # Check if image has reasonable dimensions
                width, height = img.size
                if width < 10 or height < 10:
                    issues.append(f"Symbol {name} has very small dimensions: {width}x{height}")
                
                if width > 5000 or height > 5000:
                    issues.append(f"Symbol {name} has very large dimensions: {width}x{height}")
                
            except Exception as e:
                issues.append(f"Could not load symbol {name}: {e}")
        
        return issues 