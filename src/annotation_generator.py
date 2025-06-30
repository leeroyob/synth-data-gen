"""
Annotation generator for creating ML training annotations.
Converts placement information into JSON format compatible with ML frameworks.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from placement_engine import PlacementInfo


class AnnotationGenerator:
    """Generates annotations for ML training from symbol placements."""
    
    def __init__(self, image_width: int, image_height: int):
        """
        Initialize the annotation generator.
        
        Args:
            image_width: Width of the generated image in pixels
            image_height: Height of the generated image in pixels
        """
        self.image_width = image_width
        self.image_height = image_height
    
    def create_annotation(self, image_filename: str, 
                         placements: List[PlacementInfo],
                         include_metadata: bool = True) -> Dict[str, Any]:
        """
        Create a complete annotation for an image.
        
        Args:
            image_filename: Name of the image file
            placements: List of symbol placements
            include_metadata: Whether to include metadata in the annotation
            
        Returns:
            Complete annotation dictionary
        """
        annotation = {
            "image": {
                "filename": image_filename,
                "width": self.image_width,
                "height": self.image_height
            },
            "annotations": []
        }
        
        # Add metadata if requested
        if include_metadata:
            annotation["metadata"] = {
                "created_at": datetime.now().isoformat(),
                "total_symbols": len(placements),
                "generator": "synthetic-data-generator",
                "version": "1.0"
            }
        
        # Convert placements to annotations
        for placement in placements:
            annotation_obj = self._placement_to_annotation(placement)
            annotation["annotations"].append(annotation_obj)
        
        return annotation
    
    def _placement_to_annotation(self, placement: PlacementInfo) -> Dict[str, Any]:
        """
        Convert a placement to an annotation object.
        
        Args:
            placement: PlacementInfo object
            
        Returns:
            Annotation dictionary
        """
        # Basic bounding box annotation
        annotation = {
            "class": placement.class_name,
            "bbox": {
                "x": placement.x,
                "y": placement.y,
                "width": placement.width,
                "height": placement.height
            },
            "center": {
                "x": placement.x + placement.width // 2,
                "y": placement.y + placement.height // 2
            }
        }
        
        # Add transformation information
        if placement.rotation != 0:
            annotation["rotation"] = placement.rotation
        
        if placement.scale != 1.0:
            annotation["scale"] = placement.scale
        
        if placement.flipped_h or placement.flipped_v:
            annotation["flipped"] = {
                "horizontal": placement.flipped_h,
                "vertical": placement.flipped_v
            }
        
        # Add symbol name for reference
        annotation["symbol_name"] = placement.symbol_name
        
        return annotation
    
    def create_yolo_annotation(self, placements: List[PlacementInfo],
                              class_mapping: Dict[str, int]) -> List[str]:
        """
        Create YOLO format annotations.
        
        Args:
            placements: List of symbol placements
            class_mapping: Dictionary mapping class names to class IDs
            
        Returns:
            List of YOLO format annotation strings
        """
        yolo_annotations = []
        
        for placement in placements:
            class_id = class_mapping.get(placement.class_name)
            if class_id is None:
                continue  # Skip unknown classes
            
            # Convert to YOLO format (normalized coordinates)
            center_x = (placement.x + placement.width / 2) / self.image_width
            center_y = (placement.y + placement.height / 2) / self.image_height
            width = placement.width / self.image_width
            height = placement.height / self.image_height
            
            # YOLO format: class_id center_x center_y width height
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            yolo_annotations.append(yolo_line)
        
        return yolo_annotations
    
    def create_coco_annotation(self, image_id: int, image_filename: str,
                              placements: List[PlacementInfo],
                              class_mapping: Dict[str, int],
                              start_annotation_id: int = 1) -> Dict[str, Any]:
        """
        Create COCO format annotations.
        
        Args:
            image_id: Unique image ID
            image_filename: Name of the image file
            placements: List of symbol placements
            class_mapping: Dictionary mapping class names to class IDs
            start_annotation_id: Starting ID for annotations
            
        Returns:
            COCO format annotation dictionary
        """
        # Image info
        image_info = {
            "id": image_id,
            "file_name": image_filename,
            "width": self.image_width,
            "height": self.image_height
        }
        
        # Annotations
        annotations = []
        annotation_id = start_annotation_id
        
        for placement in placements:
            class_id = class_mapping.get(placement.class_name)
            if class_id is None:
                continue  # Skip unknown classes
            
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [placement.x, placement.y, placement.width, placement.height],
                "area": placement.width * placement.height,
                "iscrowd": 0
            }
            
            annotations.append(annotation)
            annotation_id += 1
        
        return {
            "image": image_info,
            "annotations": annotations
        }
    
    def create_class_mapping(self, placements: List[PlacementInfo]) -> Dict[str, int]:
        """
        Create a mapping from class names to integer IDs.
        
        Args:
            placements: List of placements to extract classes from
            
        Returns:
            Dictionary mapping class names to IDs
        """
        unique_classes = set(placement.class_name for placement in placements)
        return {class_name: idx for idx, class_name in enumerate(sorted(unique_classes))}
    
    def save_annotation(self, annotation: Dict[str, Any], 
                       output_path: str, format: str = "json") -> None:
        """
        Save annotation to file.
        
        Args:
            annotation: Annotation dictionary
            output_path: Path to save the annotation
            format: Format to save in ('json', 'yolo', 'coco')
        """
        output_file = Path(output_path)
        
        if format.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump(annotation, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_yolo_annotation(self, yolo_annotations: List[str], 
                           output_path: str) -> None:
        """
        Save YOLO format annotations to file.
        
        Args:
            yolo_annotations: List of YOLO format strings
            output_path: Path to save the annotations
        """
        with open(output_path, 'w') as f:
            for annotation in yolo_annotations:
                f.write(annotation + '\n')
    
    def create_class_names_file(self, class_mapping: Dict[str, int], 
                               output_path: str) -> None:
        """
        Create a class names file for YOLO training.
        
        Args:
            class_mapping: Dictionary mapping class names to IDs
            output_path: Path to save the class names file
        """
        # Sort by class ID to ensure correct order
        sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
        
        with open(output_path, 'w') as f:
            for class_name, _ in sorted_classes:
                f.write(class_name + '\n')
    
    def validate_annotation(self, annotation: Dict[str, Any]) -> List[str]:
        """
        Validate an annotation and return any issues found.
        
        Args:
            annotation: Annotation dictionary to validate
            
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Check required fields
        if "image" not in annotation:
            issues.append("Missing 'image' field")
        else:
            image_info = annotation["image"]
            required_image_fields = ["filename", "width", "height"]
            for field in required_image_fields:
                if field not in image_info:
                    issues.append(f"Missing image field: {field}")
        
        if "annotations" not in annotation:
            issues.append("Missing 'annotations' field")
        else:
            for i, ann in enumerate(annotation["annotations"]):
                # Check required annotation fields
                required_ann_fields = ["class", "bbox"]
                for field in required_ann_fields:
                    if field not in ann:
                        issues.append(f"Annotation {i}: Missing field '{field}'")
                
                # Check bbox format
                if "bbox" in ann:
                    bbox = ann["bbox"]
                    required_bbox_fields = ["x", "y", "width", "height"]
                    for field in required_bbox_fields:
                        if field not in bbox:
                            issues.append(f"Annotation {i}: Missing bbox field '{field}'")
                    
                    # Check bbox values are within image bounds
                    if all(field in bbox for field in required_bbox_fields):
                        if (bbox["x"] < 0 or bbox["y"] < 0 or 
                            bbox["x"] + bbox["width"] > self.image_width or
                            bbox["y"] + bbox["height"] > self.image_height):
                            issues.append(f"Annotation {i}: Bbox outside image bounds")
        
        return issues
    
    def get_annotation_statistics(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about an annotation.
        
        Args:
            annotation: Annotation dictionary
            
        Returns:
            Statistics dictionary
        """
        if "annotations" not in annotation:
            return {"error": "No annotations found"}
        
        annotations = annotation["annotations"]
        
        # Count by class
        class_counts = {}
        total_area = 0
        
        for ann in annotations:
            class_name = ann.get("class", "unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if "bbox" in ann:
                bbox = ann["bbox"]
                area = bbox.get("width", 0) * bbox.get("height", 0)
                total_area += area
        
        # Calculate coverage
        image_area = self.image_width * self.image_height
        coverage = total_area / image_area if image_area > 0 else 0
        
        return {
            "total_annotations": len(annotations),
            "unique_classes": len(class_counts),
            "class_distribution": class_counts,
            "total_area": total_area,
            "coverage": coverage
        } 