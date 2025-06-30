"""
Configuration loader for the synthetic data generator.
Handles loading and validation of configuration files.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Loads and validates configuration for the synthetic data generator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to the configuration file. If None, uses default.
        """
        if config_path is None:
            self.config_path = Path(__file__).parent.parent / "config" / "config.json"
        else:
            self.config_path = Path(config_path)
        self.config = None
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the JSON file.
        
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
            ValueError: If config validation fails
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        self._validate_config()
        return self.config
    
    def _validate_config(self) -> None:
        """
        Validate the loaded configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config:
            raise ValueError("Configuration is empty")
        
        # Check required top-level sections
        required_sections = ['generation', 'symbol_placement', 'distortions', 'output', 'paths']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate generation section
        gen_config = self.config['generation']
        if gen_config['num_images'] <= 0:
            raise ValueError("num_images must be positive")
        if gen_config['dpi'] <= 0:
            raise ValueError("dpi must be positive")
        if len(gen_config['sheet_size_inches']) != 2:
            raise ValueError("sheet_size_inches must be [width, height]")
        if gen_config['margins_inches'] < 0:
            raise ValueError("margins_inches must be non-negative")
        
        # Validate symbol placement
        placement_config = self.config['symbol_placement']
        density_range = placement_config['density_range']
        if len(density_range) != 2 or density_range[0] > density_range[1]:
            raise ValueError("density_range must be [min, max] with min <= max")
        
        # Validate paths exist
        paths_config = self.config['paths']
        base_path = self.config_path.parent.parent
        
        for path_key, path_value in paths_config.items():
            full_path = base_path / path_value
            if path_key == 'output_dir':
                # Create output directory if it doesn't exist
                full_path.mkdir(parents=True, exist_ok=True)
            elif not full_path.exists():
                print(f"Warning: Path {path_key} does not exist: {full_path}")
    
    def get_sheet_size_pixels(self) -> tuple[int, int]:
        """
        Get sheet size in pixels based on DPI and inch dimensions.
        
        Returns:
            Tuple of (width_px, height_px)
        """
        if not self.config:
            raise ValueError("Configuration not loaded")
        
        width_inches, height_inches = self.config['generation']['sheet_size_inches']
        dpi = self.config['generation']['dpi']
        
        return (int(width_inches * dpi), int(height_inches * dpi))
    
    def get_placement_bounds(self) -> tuple[int, int, int, int]:
        """
        Get placement bounds in pixels (x, y, width, height).
        
        Returns:
            Tuple of (x, y, width, height) for symbol placement area
        """
        if not self.config:
            raise ValueError("Configuration not loaded")
        
        sheet_width_px, sheet_height_px = self.get_sheet_size_pixels()
        dpi = self.config['generation']['dpi']
        margin_inches = self.config['generation']['margins_inches']
        margin_px = int(margin_inches * dpi)
        
        # Calculate placement area (with margins, bottom half only)
        x = margin_px
        y = sheet_height_px // 2  # Bottom half only
        width = sheet_width_px - 2 * margin_px
        height = sheet_height_px // 2 - margin_px
        
        return (x, y, width, height)
    
    def get_distortion_config(self, distortion_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific distortion type.
        
        Args:
            distortion_type: Type of distortion (e.g., 'rotation', 'scaling')
            
        Returns:
            Configuration dictionary for the distortion
        """
        if not self.config:
            raise ValueError("Configuration not loaded")
        
        return self.config['distortions'].get(distortion_type, {})
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to a file.
        
        Args:
            output_path: Path to save the config. If None, overwrites original.
        """
        if not self.config:
            raise ValueError("No configuration to save")
        
        save_path = Path(output_path) if output_path else self.config_path
        
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)


# Convenience function for quick config loading
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.load_config() 