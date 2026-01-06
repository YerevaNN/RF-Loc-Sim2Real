"""
Scene Generator for Sionna RT simulation with robust PLY mesh generation.
Creates XML scenes with proper PLY mesh files for OSM building data.
"""

import os
import numpy as np
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Tuple, Optional
import logging
import re
import random
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)


def extract_height_from_osm(building):
    """Extract building height from OSM data with multiple fallback strategies"""
    

    def parse_height_string(height_str):
        if not height_str:
            return None
        # Convert to string and clean
        height_str = str(height_str).strip().lower()
        # Extract number and unit using regex
        match = re.match(r'^(\d*\.?\d+)\s*(m|meters?|ft|feet|\')?', height_str)
        if match:
            value = float(match.group(1))
            unit = match.group(2) or 'm'  # Default to meters
            # Convert feet to meters
            if unit in ['ft', 'feet', "'"]:
                value = value * 0.3048
            return value
        return None
    
    # Method 1: Direct height attribute
    for height_key in ['height', 'building:height', 'roof:height']:
        if height_key in building and building[height_key]:
            height = parse_height_string(building[height_key])
            if height and 2.0 <= height <= 200.0:  # Reasonable bounds
                return height
    
    # Method 2: Building levels
    for levels_key in ['building:levels', 'levels']:
        if levels_key in building and building[levels_key]:
            try:
                levels = float(building[levels_key])
                if 1 <= levels <= 50:  # Reasonable bounds
                    # Vary floor height by building type
                    building_type = building.get('building', 'yes')
                    if building_type in ['industrial', 'warehouse']:
                        floor_height = 4.5  # Higher ceilings
                    elif building_type in ['commercial', 'office', 'retail']:
                        floor_height = 3.5  # Standard commercial
                    else:
                        floor_height = 3.0  # Residential
                    
                    return levels * floor_height
            except (ValueError, TypeError):
                pass
    
    # Method 3: Building type-specific defaults with variation
    building_type = building.get('building', 'yes')
    amenity = building.get('amenity', '')
    
    # Create realistic height ranges by type
    if building_type in ['house', 'residential', 'detached', 'semi_detached']:
        base_height = random.uniform(5.5, 8.5)
    elif building_type in ['apartments', 'dormitory']:
        base_height = random.uniform(12.0, 25.0)
    elif building_type in ['commercial', 'office', 'retail']:
        base_height = random.uniform(10.0, 20.0)
    elif building_type in ['industrial', 'warehouse', 'manufacture']:
        base_height = random.uniform(6.0, 12.0)
    elif building_type in ['church', 'cathedral', 'chapel']:
        base_height = random.uniform(15.0, 35.0)
    elif building_type in ['school', 'university']:
        base_height = random.uniform(8.0, 15.0)
    elif building_type in ['hospital']:
        base_height = random.uniform(15.0, 30.0)
    elif amenity in ['place_of_worship']:
        base_height = random.uniform(12.0, 25.0)
    elif amenity in ['school', 'university']:
        base_height = random.uniform(8.0, 15.0)
    else:
        # Generic building - vary by footprint size if available
        try:
            area = building['geometry'].area * 111320 * 111320  # Rough area in m²
            if area > 1000:  # Large building
                base_height = random.uniform(12.0, 25.0)
            elif area > 200:  # Medium building  
                base_height = random.uniform(8.0, 18.0)
            else:  # Small building
                base_height = random.uniform(5.0, 12.0)
        except:
            base_height = random.uniform(8.0, 15.0)
    
    return round(base_height, 1)


def create_ground_plane_ply(
    top: float,
    bottom: float,
    left: float,
    right: float,
    *,
    margin: float | None = None,        # metres to add on every side (e.g. 20.0)
    margin_pct: float | None = 0.0,     # percentage of width/height to add (0.0 = same size)
    z_level: float = -0.5,              # push plane slightly below 0
    output_dir: str = "meshes",
    filename: str = "Plane.ply",
) -> str | None:
    """
    Create a PLY ground plane to cover the scene.
    
    Either `margin` **or** `margin_pct` can be given. If both are `None`,
    no expansion is applied.
    """
    
    # 1. Determine expansion amount
    width = right - left
    height = bottom - top
    
    if margin is not None:
        # fixed expansion in metres
        expand_x = expand_y = float(margin)
    else:
        # percentage-based expansion (default 0% = same size)
        expand_x = width * float(margin_pct or 0)
        expand_y = height * float(margin_pct or 0)
    
    # 2. Compute enlarged bounding box
    ext_left = left - expand_x
    ext_right = right + expand_x
    ext_top = top - expand_y
    ext_bottom = bottom + expand_y
    
    # Plane vertices (counter-clockwise order)
    vertices = [
        (ext_top, ext_left, z_level),
        (ext_top, ext_right, z_level),
        (ext_bottom, ext_right, z_level),
        (ext_bottom, ext_left, z_level),
    ]
    
    # Two triangles → one quad
    faces = [
        (np.array([0, 1, 2], dtype='i4'),),
        (np.array([0, 2, 3], dtype='i4'),),
    ]
    
    # 3. Write PLY
    try:
        vertex_array = np.array(vertices,
                               dtype=[('x', 'f4'),
                                      ('y', 'f4'),
                                      ('z', 'f4')])
        face_array = np.array(faces,
                             dtype=[('vertex_indices', 'i4', (3,))])
        
        ply_data = PlyData([
            PlyElement.describe(vertex_array, 'vertex'),
            PlyElement.describe(face_array, 'face')
        ], text=True)
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        ply_data.write(output_path)
        return output_path
    
    except Exception as e:
        logger.error(f"[create_ground_plane_ply] Error: {e}")
        return None


def triangulate_polygon(polygon_coords):
    """Simple ear clipping triangulation for convex polygons with validation"""
    triangles = []
    
    # Make sure we remove the last point if it's the same as the first
    coords = polygon_coords.copy()
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords.pop()
    
    n = len(coords)
    
    if n < 3:
        return triangles  # Not enough points to form a triangle
    
    # Create triangles by connecting first point to pairs of consecutive vertices
    for i in range(1, n - 1):
        try:
            # Validate that all coordinates can be converted to float
            p0 = (float(coords[0][0]), float(coords[0][1]), 0.0)
            p1 = (float(coords[i][0]), float(coords[i][1]), 0.0)
            p2 = (float(coords[i+1][0]), float(coords[i+1][1]), 0.0)
            
            # Check for NaN or infinite values
            if (math.isnan(p0[0]) or math.isnan(p0[1]) or
                math.isnan(p1[0]) or math.isnan(p1[1]) or
                math.isnan(p2[0]) or math.isnan(p2[1]) or
                math.isinf(p0[0]) or math.isinf(p0[1]) or
                math.isinf(p1[0]) or math.isinf(p1[1]) or
                math.isinf(p2[0]) or math.isinf(p2[1])):
                continue
            
            triangles.append([p0, p1, p2])
        except (ValueError, TypeError):
            # Skip this triangle if coordinates can't be converted to float
            continue
    
    return triangles


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


class SceneGenerator:
    """Generates complete Sionna RT scenes with buildings from OSM data."""
    
    def __init__(self, output_dir: str):
        """
        Initialize scene generator.
        
        Args:
            output_dir: Directory to save generated files
        """
        self.output_dir = output_dir
        self.meshes_dir = os.path.join(output_dir, "meshes")
        os.makedirs(self.meshes_dir, exist_ok=True)
    
    def generate_complete_scene(self, building_geometries: List[Dict], 
                              bounds: Dict[str, Tuple[float, float]], 
                              center_point: Tuple[float, float],
                              scene_name: str,
                              ground_margin_pct: float = 0.0) -> Tuple[str, str]:
        """
        Generate complete scene with XML and PLY mesh files.
        
        Args:
            building_geometries: List of building geometry dictionaries
            bounds: Region bounds
            center_point: Center lat/lon coordinates  
            scene_name: Name for the scene
            ground_margin_pct: Percentage margin for ground plane (0.0 = same size as buildings)
            
        Returns:
            Tuple of (xml_path, meshes_dir)
        """
        logger.info(f"Generating complete scene: {scene_name}")
        logger.info(f"Building geometries: {len(building_geometries)}")
        
        # Generate PLY mesh files for buildings
        ply_files = self._generate_building_meshes(building_geometries)
        
        # Generate ground plane PLY with specified margin
        ground_ply = self._generate_ground_plane(bounds, center_point, ground_margin_pct)
        if ground_ply:
            ply_files.append(ground_ply)
        
        # Generate XML scene file
        xml_path = self._generate_scene_xml([f for f in ply_files if f is not None], scene_name)
        
        logger.info(f"Scene generation completed:")
        logger.info(f"  XML: {xml_path}")
        logger.info(f"  Meshes: {self.meshes_dir} ({len(ply_files)} files)")
        
        return xml_path, self.meshes_dir
    
    def _generate_building_meshes(self, building_geometries: List[Dict]) -> List[str]:
        """Generate PLY mesh files for building geometries."""
        ply_files = []
        
        logger.info(f"Generating {len(building_geometries)} building meshes...")
        
        for i, building in enumerate(building_geometries):
            try:
                building_id = building.get('id', f'building_{i}')
                ply_filename = f"{building_id}.ply"
                ply_path = os.path.join(self.meshes_dir, ply_filename)
                
                # Generate building mesh
                success = self._create_building_ply(building, ply_path)
                
                if success:
                    ply_files.append(ply_filename)
                    logger.debug(f"Generated mesh: {ply_filename}")
                else:
                    logger.warning(f"Failed to generate mesh for building {building_id}")
                    
            except Exception as e:
                logger.error(f"Error generating mesh for building {i}: {e}")
        
        logger.info(f"Successfully generated {len(ply_files)} building meshes")
        return ply_files
    
    def _create_building_ply(self, building: Dict, ply_path: str) -> bool:
        """Create a PLY file for a single building using the proven working method."""
        try:
            exterior = building['exterior']
            height = building.get('height', 10.0)
            
            if len(exterior) < 3:
                logger.warning(f"Building has insufficient vertices: {len(exterior)}")
                return False
            
            # Use the proven working mesh creation method
            mesh = self._create_building_mesh_working(building)
            
            # Collect all vertices and faces
            vertices = []
            faces = []
            vertex_count = 0
            
            # Process each section (walls, roof, ground)
            for section in ['walls', 'roof', 'ground']:
                for triangle in mesh[section]:
                    # Validate and clean vertices before adding
                    valid_vertices = []
                    for vertex in triangle:
                        try:
                            # Ensure each coordinate is a valid float
                            x = float(vertex[0])
                            y = float(vertex[1])
                            z = float(vertex[2])
                            
                            # Check for NaN or infinite values
                            if (math.isnan(x) or math.isnan(y) or math.isnan(z) or
                                math.isinf(x) or math.isinf(y) or math.isinf(z)):
                                continue
                            
                            valid_vertices.append((x, y, z))
                        except (ValueError, TypeError):
                            continue
                    
                    # Only add face if we have exactly 3 valid vertices
                    if len(valid_vertices) == 3:
                        # Add vertices
                        for v in valid_vertices:
                            vertices.append(v)
                        
                        # Add face (triangle)
                        faces.append((np.array([vertex_count, vertex_count+1, vertex_count+2], dtype='i4'),))
                        vertex_count += 3
            
            # Skip if no valid vertices
            if len(vertices) == 0:
                logger.warning(f"No valid vertices for building")
                return False
            
            # Create PLY data structure
            vertex_data = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            vertex_element = PlyElement.describe(vertex_data, 'vertex')
            
            face_data = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
            face_element = PlyElement.describe(face_data, 'face')
            
            # Create PLY data and save to file
            ply_data = PlyData([vertex_element, face_element], text=True)
            ply_data.write(ply_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating PLY file {ply_path}: {e}")
            return False
    
    def _create_building_mesh_working(self, building):
        """Create a 3D mesh for a building by extruding its footprint - working version"""
        exterior = building['exterior']
        try:
            height = float(building['height'])
            # Ensure height is valid
            if math.isnan(height) or math.isinf(height) or height <= 0:
                height = 10.0  # Use default if invalid
        except (ValueError, TypeError):
            height = 10.0  # Use default if conversion fails
        
        # Remove the last point if it's the same as the first (closed polygon)
        if len(exterior) > 1 and exterior[0] == exterior[-1]:
            exterior = exterior[:-1]
        
        # Ensure we have at least 3 points to form a polygon
        if len(exterior) < 3:
            return {'ground': [], 'roof': [], 'walls': []}
        
        # Ground face triangulation - explicitly add z=0 to make 3D points
        ground_triangles = triangulate_polygon(exterior)
        
        # Roof face triangulation (same as ground but elevated)
        roof_triangles = []
        for triangle in ground_triangles:
            try:
                roof_triangle = [
                    (float(p[0]), float(p[1]), height) for p in triangle
                ]
                # Verify all points are valid
                valid = True
                for p in roof_triangle:
                    if (math.isnan(p[0]) or math.isnan(p[1]) or math.isnan(p[2]) or
                        math.isinf(p[0]) or math.isinf(p[1]) or math.isinf(p[2])):
                        valid = False
                        break
                if valid:
                    roof_triangles.append(roof_triangle)
            except (ValueError, TypeError):
                continue
        
        # Wall faces (connect ground and roof edges)
        wall_triangles = []
        for i in range(len(exterior)):
            try:
                p1 = exterior[i]
                p2 = exterior[(i+1) % len(exterior)]
                
                # Create 2 triangles for each rectangular wall face
                # Explicitly add z coordinates to make 3D points
                g1 = (float(p1[0]), float(p1[1]), 0.0)
                g2 = (float(p2[0]), float(p2[1]), 0.0)
                r1 = (float(p1[0]), float(p1[1]), height)
                r2 = (float(p2[0]), float(p2[1]), height)
                
                # Verify coordinates
                all_points = [g1, g2, r1, r2]
                valid = True
                for p in all_points:
                    if (math.isnan(p[0]) or math.isnan(p[1]) or math.isnan(p[2]) or
                        math.isinf(p[0]) or math.isinf(p[1]) or math.isinf(p[2])):
                        valid = False
                        break
                
                if valid:
                    wall_triangles.append([g1, g2, r1])
                    wall_triangles.append([g2, r2, r1])
            except (ValueError, TypeError):
                continue
        
        # Combine all triangles
        all_triangles = {
            'ground': ground_triangles,
            'roof': roof_triangles,
            'walls': wall_triangles
        }
        
        return all_triangles
    
    def _generate_ground_plane(self, bounds: Dict[str, Tuple[float, float]], 
                             center_point: Tuple[float, float],
                             margin_pct: float = 0.0) -> str:
        """Generate a ground plane PLY file using the working method."""
        logger.info(f"Generating ground plane mesh with {margin_pct*100}% margin...")
        
        # Convert bounds to approximate local coordinates for plane sizing
        lat_range = bounds['lat_range']
        lon_range = bounds['lon_range']
        
        # Convert approximate bounds to local coordinates
        # This is a rough approximation - in real implementation you'd use the same
        # coordinate transformation as used for buildings
        lat_center = (lat_range[0] + lat_range[1]) / 2
        lon_center = (lon_range[0] + lon_range[1]) / 2
        
        # Rough coordinate bounds (this should match your lonlat_to_local function)
        top = (lat_range[1] - lat_center) * 111000
        bottom = (lat_range[0] - lat_center) * 111000
        left = (lon_range[0] - lon_center) * 111000 * np.cos(np.radians(lat_center))
        right = (lon_range[1] - lon_center) * 111000 * np.cos(np.radians(lat_center))
        
        # Use the working ground plane creation function
        ply_filename = "Plane.ply"
        ground_file = create_ground_plane_ply(
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            margin_pct=margin_pct,
            z_level=-0.5,
            output_dir=self.meshes_dir,
            filename=ply_filename
        )
        
        if ground_file:
            logger.info(f"Generated ground plane: {ply_filename}")
            return ply_filename
        else:
            logger.error("Failed to generate ground plane")
            return None
    
    def _generate_scene_xml(self, ply_files: List[str], scene_name: str) -> str:
        """Generate XML scene file using the proven working structure."""
        logger.info(f"Generating XML scene file: {scene_name}.xml")
        
        # Create the root element - use working version
        scene = ET.Element("scene", version="2.1.0")
        
        # Add a comment
        scene.append(ET.Comment(" Camera and Rendering Parameters "))
        
        # Add integrator
        integrator = ET.SubElement(scene, "integrator", type="path", id="elm__0", name="elm__0")
        ET.SubElement(integrator, "integer", name="max_depth", value="12")
        
        # Add materials section
        scene.append(ET.Comment(" Materials "))
        
        # Add concrete material
        bsdf = ET.SubElement(scene, "bsdf", type="twosided", id="mat-itu_concrete")
        bsdf_diffuse = ET.SubElement(bsdf, "bsdf", type="diffuse")
        ET.SubElement(bsdf_diffuse, "rgb", value="0.539479 0.539479 0.539480", name="reflectance")
        
        # Add emitters section
        scene.append(ET.Comment(" Emitters "))
        
        # Add shapes section
        scene.append(ET.Comment(" Shapes "))
        
        # Add ground plane reference
        if "Plane.ply" in ply_files:
            ground = ET.SubElement(scene, "shape", type="ply", id="elm__2", name="elm__2")
            ET.SubElement(ground, "string", name="filename", value="meshes/Plane.ply")
            ET.SubElement(ground, "ref", id="mat-itu_concrete", name="bsdf")
        
        # Add building references
        building_ply_files = [f for f in ply_files if f != 'Plane.ply' and f is not None]
        for i, building_id in enumerate(building_ply_files):
            building = ET.SubElement(scene, "shape", type="ply",
                                   id=f"elm__{i+3}", name=f"elm__{i+3}")
            ET.SubElement(building, "string", name="filename",
                         value=f"meshes/{building_id}")
            ET.SubElement(building, "ref", id="mat-itu_concrete", name="bsdf")
        
        # Add volumes section
        scene.append(ET.Comment(" Volumes "))
        
        # Write XML file
        xml_path = os.path.join(self.output_dir, f"{scene_name}.xml")
        
        # Use the working prettify function
        pretty_xml = prettify(scene)
        
        with open(xml_path, 'w') as f:
            f.write(pretty_xml)
        
        logger.info(f"Generated XML scene: {xml_path}")
        logger.info(f"Added {len(building_ply_files)} building shapes to scene")
        
        return xml_path