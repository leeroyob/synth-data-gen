"""
Placement engine for positioning symbols on engineering sheets.
Handles random placement, collision detection, and bounding box tracking.
"""

import random
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image


@dataclass
class PlacementInfo:
    """Information about a placed symbol."""
    symbol_name: str
    class_name: str
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int
    height: int
    rotation: float = 0.0
    scale: float = 1.0
    flipped_h: bool = False
    flipped_v: bool = False
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of the placement."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def overlaps_with(self, other: 'PlacementInfo', padding: int = 0) -> bool:
        """Check if this placement overlaps with another."""
        # Expand bounding boxes by padding
        x1, y1, w1, h1 = self.get_bbox()
        x2, y2, w2, h2 = other.get_bbox()
        
        x1 -= padding
        y1 -= padding
        w1 += 2 * padding
        h1 += 2 * padding
        
        x2 -= padding
        y2 -= padding
        w2 += 2 * padding
        h2 += 2 * padding
        
        # Check for overlap
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


class PlacementEngine:
    """Handles placement of symbols on engineering sheets."""
    
    def __init__(self, placement_bounds: Tuple[int, int, int, int]):
        """
        Initialize the placement engine.
        
        Args:
            placement_bounds: (x, y, width, height) defining the placement area
        """
        self.bounds_x, self.bounds_y, self.bounds_width, self.bounds_height = placement_bounds
        self.placements: List[PlacementInfo] = []
        self.collision_padding = 10  # Default padding for collision detection
        self.max_placement_attempts = 100  # Maximum attempts to place a symbol
    
    def set_collision_padding(self, padding: int) -> None:
        """Set the padding for collision detection."""
        self.collision_padding = padding
    
    def clear_placements(self) -> None:
        """Clear all current placements."""
        self.placements.clear()
    
    def get_placements(self) -> List[PlacementInfo]:
        """Get all current placements."""
        return self.placements.copy()
    
    def _is_within_bounds(self, x: int, y: int, width: int, height: int) -> bool:
        """Check if a placement is within the allowed bounds."""
        return (x >= self.bounds_x and 
                y >= self.bounds_y and 
                x + width <= self.bounds_x + self.bounds_width and 
                y + height <= self.bounds_y + self.bounds_height)
    
    def _has_collision(self, placement: PlacementInfo, check_collisions: bool = True) -> bool:
        """Check if a placement would collide with existing placements."""
        if not check_collisions:
            return False
        
        for existing in self.placements:
            if placement.overlaps_with(existing, self.collision_padding):
                return True
        return False
    
    def _calculate_symbol_dimensions(self, symbol_size: Tuple[int, int], 
                                   scale: float, rotation: float) -> Tuple[int, int]:
        """
        Calculate the dimensions of a symbol after scaling and rotation.
        
        Args:
            symbol_size: Original (width, height) of the symbol
            scale: Scale factor
            rotation: Rotation angle in degrees
            
        Returns:
            Tuple of (width, height) after transformations
        """
        width, height = symbol_size
        
        # Apply scaling
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)
        
        # Calculate bounding box after rotation
        if rotation != 0:
            rad = math.radians(abs(rotation))
            cos_r = math.cos(rad)
            sin_r = math.sin(rad)
            
            # Calculate the bounding box of the rotated rectangle
            new_width = int(abs(scaled_width * cos_r) + abs(scaled_height * sin_r))
            new_height = int(abs(scaled_width * sin_r) + abs(scaled_height * cos_r))
            
            return (new_width, new_height)
        
        return (scaled_width, scaled_height)
    
    def place_symbol_random(self, symbol_name: str, class_name: str, 
                          symbol_size: Tuple[int, int],
                          scale_range: Tuple[float, float] = (1.0, 1.0),
                          rotation_range: Tuple[float, float] = (0.0, 0.0),
                          allow_flip: bool = False,
                          check_collisions: bool = True) -> Optional[PlacementInfo]:
        """
        Place a symbol at a random location within bounds.
        
        Args:
            symbol_name: Name of the symbol
            class_name: Class name for the symbol
            symbol_size: Original (width, height) of the symbol
            scale_range: (min_scale, max_scale) for random scaling
            rotation_range: (min_rotation, max_rotation) in degrees
            allow_flip: Whether to allow random flipping
            check_collisions: Whether to check for collisions
            
        Returns:
            PlacementInfo if successful, None if placement failed
        """
        for attempt in range(self.max_placement_attempts):
            # Generate random transformations
            scale = random.uniform(*scale_range)
            rotation = random.uniform(*rotation_range)
            flip_h = random.choice([True, False]) if allow_flip else False
            flip_v = random.choice([True, False]) if allow_flip else False
            
            # Calculate final dimensions
            final_width, final_height = self._calculate_symbol_dimensions(
                symbol_size, scale, rotation
            )
            
            # Generate random position within bounds
            max_x = self.bounds_x + self.bounds_width - final_width
            max_y = self.bounds_y + self.bounds_height - final_height
            
            if max_x < self.bounds_x or max_y < self.bounds_y:
                # Symbol is too large for the bounds
                continue
            
            x = random.randint(self.bounds_x, max_x)
            y = random.randint(self.bounds_y, max_y)
            
            # Create placement info
            placement = PlacementInfo(
                symbol_name=symbol_name,
                class_name=class_name,
                x=x, y=y,
                width=final_width,
                height=final_height,
                rotation=rotation,
                scale=scale,
                flipped_h=flip_h,
                flipped_v=flip_v
            )
            
            # Check if placement is valid
            if (self._is_within_bounds(x, y, final_width, final_height) and 
                not self._has_collision(placement, check_collisions)):
                
                self.placements.append(placement)
                return placement
        
        # Could not place symbol after max attempts
        return None
    
    def place_symbols_batch(self, symbols_data: List[Dict[str, Any]], 
                          placement_config: Dict[str, Any]) -> List[PlacementInfo]:
        """
        Place multiple symbols according to configuration.
        
        Args:
            symbols_data: List of symbol dictionaries with 'name', 'class_name', 'size'
            placement_config: Configuration for placement parameters
            
        Returns:
            List of successful placements
        """
        successful_placements = []
        
        # Extract configuration
        scale_range = placement_config.get('scale_range', (1.0, 1.0))
        rotation_range = placement_config.get('rotation_range', (0.0, 360.0))
        allow_flip = placement_config.get('allow_flip', False)
        check_collisions = placement_config.get('check_collisions', True)
        
        for symbol_data in symbols_data:
            placement = self.place_symbol_random(
                symbol_name=symbol_data['name'],
                class_name=symbol_data['class_name'],
                symbol_size=symbol_data['size'],
                scale_range=scale_range,
                rotation_range=rotation_range,
                allow_flip=allow_flip,
                check_collisions=check_collisions
            )
            
            if placement:
                successful_placements.append(placement)
        
        return successful_placements
    
    def generate_placement_grid(self, num_symbols: int, 
                              symbol_size: Tuple[int, int],
                              spacing: int = 50) -> List[Tuple[int, int]]:
        """
        Generate a grid of positions for systematic placement.
        
        Args:
            num_symbols: Number of positions to generate
            symbol_size: Size of symbols for spacing calculation
            spacing: Minimum spacing between positions
            
        Returns:
            List of (x, y) positions
        """
        positions = []
        symbol_width, symbol_height = symbol_size
        
        # Calculate grid dimensions
        total_width = self.bounds_width
        total_height = self.bounds_height
        
        cols = max(1, total_width // (symbol_width + spacing))
        rows = max(1, total_height // (symbol_height + spacing))
        
        # Generate grid positions
        for row in range(rows):
            for col in range(cols):
                if len(positions) >= num_symbols:
                    break
                
                x = self.bounds_x + col * (symbol_width + spacing)
                y = self.bounds_y + row * (symbol_height + spacing)
                
                # Ensure position is within bounds
                if (x + symbol_width <= self.bounds_x + self.bounds_width and
                    y + symbol_height <= self.bounds_y + self.bounds_height):
                    positions.append((x, y))
            
            if len(positions) >= num_symbols:
                break
        
        return positions[:num_symbols]
    
    def get_placement_density(self) -> float:
        """
        Calculate the current placement density.
        
        Returns:
            Density as a ratio of occupied area to total area
        """
        if not self.placements:
            return 0.0
        
        total_area = self.bounds_width * self.bounds_height
        occupied_area = sum(p.width * p.height for p in self.placements)
        
        return min(1.0, occupied_area / total_area)
    
    def get_placement_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about current placements.
        
        Returns:
            Dictionary with placement statistics
        """
        if not self.placements:
            return {
                'total_placements': 0,
                'density': 0.0,
                'classes': {},
                'avg_scale': 0.0,
                'avg_rotation': 0.0
            }
        
        # Count by class
        class_counts = {}
        for p in self.placements:
            class_counts[p.class_name] = class_counts.get(p.class_name, 0) + 1
        
        # Calculate averages
        avg_scale = sum(p.scale for p in self.placements) / len(self.placements)
        avg_rotation = sum(p.rotation for p in self.placements) / len(self.placements)
        
        return {
            'total_placements': len(self.placements),
            'density': self.get_placement_density(),
            'classes': class_counts,
            'avg_scale': avg_scale,
            'avg_rotation': avg_rotation,
            'bounds': (self.bounds_x, self.bounds_y, self.bounds_width, self.bounds_height)
        } 