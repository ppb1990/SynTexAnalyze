import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon, Wedge
from shapely.geometry import Polygon as ShapelyPolygon


# ----------------------------
# 1. SYNTHETIC DATA GENERATION
# ----------------------------

def generate_intensity_map():
    """
    Create synthetic intensity distribution with two peaks
    Returns:
        np.ndarray: 100x100 intensity matrix (0-1 range)
    """
    x_dim, y_dim = 350, 500  # Microns
    xx, yy = np.meshgrid(np.linspace(0, x_dim, 100),
                         np.linspace(0, y_dim, 100))

    # Create Gaussian intensity peaks
    intensity = np.exp(-((xx - 150) ** 2 + (yy - 200) ** 2) / (2 * 80 ** 2))
    intensity += 0.3 * np.exp(-((xx - 280) ** 2 + (yy - 380) ** 2) / (2 * 50 ** 2))
    return intensity ** 2  # Enhanced contrast


def generate_orientations(n_seeds):
    """
    Generate synthetic crystal orientations (Euler angles)
    Args:
        n_seeds (int): Number of grains/seeds
    Returns:
        np.ndarray: (n_seeds, 3) array of Euler angles (φ1, Φ, φ2)
    """
    # For cubic symmetry, angles typically range [0, 360] but often reduced
    return np.random.uniform(low=[0, 0, 0],
                             high=[360, 180, 360],
                             size=(n_seeds, 3))


# ----------------------------
# 2. VORONOI TESSELATION SETUP
# ----------------------------

def generate_seeds(intensity, n_seeds=300, padding=75):
    """
    Generate Voronoi seeds with intensity-based probability
    Args:
        intensity (np.ndarray): 2D intensity matrix
        n_seeds (int): Target number of seeds
        padding (int): Boundary padding (microns)
    Returns:
        np.ndarray: (n_seeds, 2) array of seed coordinates
    """
    h, w = intensity.shape
    seeds = []

    while len(seeds) < n_seeds:
        # Generate candidate seed with padding
        x = np.random.uniform(-padding, 350 + padding)
        y = np.random.uniform(-padding, 500 + padding)

        # Convert to intensity array indices with clamping
        xi = int((x / (350 + 2 * padding)) * (w - 1))
        yi = int((y / (500 + 2 * padding)) * (h - 1))
        xi = np.clip(xi, 0, w - 1)
        yi = np.clip(yi, 0, h - 1)

        # Intensity-based rejection sampling
        if np.random.rand() < (1 - intensity[yi, xi]):
            seeds.append([x, y])

    return np.array(seeds)


def clip_voronoi(vor, bbox):
    """
    Clip Voronoi regions to specified bounding box
    Args:
        vor (scipy.spatial.Voronoi): Voronoi object
        bbox (tuple): (min_x, min_y, max_x, max_y)
    Returns:
        list: Clipped polygon coordinates for each region
    """
    min_x, min_y, max_x, max_y = bbox
    bbox_poly = ShapelyPolygon([(min_x, min_y), (max_x, min_y),
                                (max_x, max_y), (min_x, max_y)])

    clipped_regions = []
    for region in vor.regions:
        if not region or -1 in region:
            continue  # Skip empty/infinite regions

        # Convert vertex indices to coordinates
        vertices = [vor.vertices[i] for i in region if i != -1]
        if len(vertices) < 3:
            continue  # Skip degenerate regions

        # Clip polygon to bounding box
        poly = ShapelyPolygon(vertices)
        clipped = poly.intersection(bbox_poly)
        if clipped.is_empty:
            continue

        # Handle different geometry types
        if isinstance(clipped, ShapelyPolygon):
            clipped_regions.append(np.array(clipped.exterior.coords))
        else:  # MultiPolygon
            for geom in clipped.geoms:
                clipped_regions.append(np.array(geom.exterior.coords))

    return clipped_regions


# ----------------------------
# 1. FIBER TEXTURE ORIENTATIONS
# ----------------------------

def generate_orientations(n_seeds):
    """
    Generate fiber texture orientations with:
    - φ1 fixed at 0° (fiber axis)
    - Φ fixed at 54.74° (characteristic angle)
    - φ2 varying between 0-360°
    """
    fixed_phi1 = 0.0  # Fiber axis direction (fixed)
    fixed_Phi = 54.74  # Fixed polar angle (characteristic angle)
    phi2 = np.random.uniform(0, 360, n_seeds)

    return np.column_stack((
        np.full(n_seeds, fixed_phi1),
        np.full(n_seeds, fixed_Phi),
        phi2
    ))


def fold_angle(angle):
    """Fold angle into 0-90° range using four-fold symmetry"""
    return angle % 90  # Equivalent to angle % 90


# ----------------------------
# 2. ORIENTATION COLOR MAPPING
# ----------------------------

def create_symmetry_colormap():
    """Create cyclic colormap for four-fold symmetry (0-90°)"""
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.0, '#1f77b4'),  # Blue at 0°
        (0.25, '#2ca02c'),  # Green at 22.5°
        (0.5, '#ff7f0e'),  # Orange at 45°
        (0.75, '#d62728'),  # Red at 67.5°
        (1.0, '#1f77b4')  # Blue at 90° (same as 0°)
    ]
    return LinearSegmentedColormap.from_list('symmetry_cmap', colors)


def orientation_to_color(phi2):
    """Convert φ2 angle to symmetry color"""
    folded = fold_angle(phi2)
    normalized = folded / 90  # Scale to 0-1
    return symmetry_cmap(normalized)


# ----------------------------
# 3. MODIFIED VISUALIZATION
# ----------------------------

def add_symmetry_colorbar(ax):
    """Add color wheel for four-fold symmetry"""
    angles = np.linspace(0, 90, 91)
    for angle in angles:
        folded = fold_angle(angle)
        norm_angle = folded / 90
        color = symmetry_cmap(norm_angle)

        # Draw wedge segment
        ax.add_patch(Wedge(
            (300, 450), 40,
            angle * 4, angle * 4 + 1,  # Scale to 0-360° for display
            color=color, alpha=0.7
        ))
    ax.text(300, 400, 'φ2 Orientation (0-90°)',
            ha='center', va='top', fontsize=7)


# ----------------------------
# MAIN SIMULATION (UPDATED)
# ----------------------------

# Create custom colormap
symmetry_cmap = create_symmetry_colormap()

# Generate data (keep other functions same as previous)
intensity = generate_intensity_map()
seeds = generate_seeds(intensity, n_seeds=400, padding=75)
orientations = generate_orientations(len(seeds))

# Compute Voronoi and clip regions (same as before)
vor = Voronoi(seeds)
clipped_regions = clip_voronoi(vor, (0, 0, 350, 500))

# Create figure
fig = plt.figure(figsize=(3.5, 5), dpi=100)
ax = fig.add_subplot(111)
ax.set(xlim=(0, 350), ylim=(0, 500))
ax.axis('off')

# Plot grains with symmetry coloring
for region in clipped_regions:
    if len(region) < 3:
        continue

    centroid = np.mean(region, axis=0)
    distances = np.linalg.norm(seeds - centroid, axis=1)
    nearest_idx = np.argmin(distances)

    # Get φ2 angle and convert to color
    phi2 = orientations[nearest_idx][2]
    color = orientation_to_color(phi2)

    ax.add_patch(Polygon(region,
                         facecolor=color,
                         edgecolor='black',
                         linewidth=0.3,
                         alpha=0.85))

# Add symmetry color wheel
add_symmetry_colorbar(ax)

plt.savefig('fiber_texture_ebsd.png', dpi=100, bbox_inches='tight')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon, Wedge
from shapely.geometry import Polygon as ShapelyPolygon

# ----------------------------
# 1. FIBER TEXTURE ORIENTATIONS
# ----------------------------

def generate_orientations(n_seeds):
    """
    Generate fiber texture orientations with:
    - φ1 fixed at 0° (fiber axis)
    - Φ fixed at 54.74° (characteristic angle)
    - φ2 varying between 0-360°
    """
    fixed_phi1 = 0.0    # Fiber axis direction (fixed)
    fixed_Phi = 54.74   # Fixed polar angle (characteristic angle)
    phi2 = np.random.uniform(0, 360, n_seeds)

    return np.column_stack((
        np.full(n_seeds, fixed_phi1),
        np.full(n_seeds, fixed_Phi),
        phi2
    ))

def fold_angle(angle):
    """Fold angle into 0-90° range using four-fold symmetry"""
    return angle % 90  # Equivalent to angle % 90

# ----------------------------
# 2. ORIENTATION COLOR MAPPING
# ----------------------------

def create_symmetry_colormap():
    """Create cyclic colormap for four-fold symmetry (0-90°)"""
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.0,   '#1f77b4'),  # Blue at 0°
        (0.25,  '#2ca02c'),  # Green at 22.5°
        (0.5,   '#ff7f0e'),  # Orange at 45°
        (0.75,  '#d62728'),  # Red at 67.5°
        (1.0,   '#1f77b4')   # Blue at 90° (same as 0°)
    ]
    return LinearSegmentedColormap.from_list('symmetry_cmap', colors)

def orientation_to_color(phi2):
    """Convert φ2 angle to symmetry color"""
    folded = fold_angle(phi2)
    normalized = folded / 90  # Scale to 0-1
    return symmetry_cmap(normalized)

# ----------------------------
# 3. MODIFIED VISUALIZATION
# ----------------------------

def add_symmetry_colorbar(ax):
    """Add color wheel for four-fold symmetry"""
    angles = np.linspace(0, 90, 91)
    for angle in angles:
        folded = fold_angle(angle)
        norm_angle = folded/90
        color = symmetry_cmap(norm_angle)

        # Draw wedge segment
        ax.add_patch(Wedge(
            (300, 450), 40,
            angle*4, angle*4+1,  # Scale to 0-360° for display
            color=color, alpha=0.7
        ))
    ax.text(300, 400, 'φ2 Orientation (0-90°)',
            ha='center', va='top', fontsize=7)

# ----------------------------
# MAIN SIMULATION (UPDATED)
# ----------------------------

# Create custom colormap
symmetry_cmap = create_symmetry_colormap()

# Generate data (keep other functions same as previous)
intensity = generate_intensity_map()
seeds = generate_seeds(intensity, n_seeds=400, padding=75)
orientations = generate_orientations(len(seeds))

# Compute Voronoi and clip regions (same as before)
vor = Voronoi(seeds)
clipped_regions = clip_voronoi(vor, (0, 0, 350, 500))

# Create figure
fig = plt.figure(figsize=(3.5, 5), dpi=100)
ax = fig.add_subplot(111)
ax.set(xlim=(0, 350), ylim=(0, 500))
ax.axis('off')

# Plot grains with symmetry coloring
for region in clipped_regions:
    if len(region) < 3:
        continue

    centroid = np.mean(region, axis=0)
    distances = np.linalg.norm(seeds - centroid, axis=1)
    nearest_idx = np.argmin(distances)

    # Get φ2 angle and convert to color
    phi2 = orientations[nearest_idx][2]
    color = orientation_to_color(phi2)

    ax.add_patch(Polygon(region,
                        facecolor=color,
                        edgecolor='black',
                        linewidth=0.3,
                        alpha=0.85))

# Add symmetry color wheel
add_symmetry_colorbar(ax)

plt.savefig('fiber_texture_ebsd.png', dpi=100, bbox_inches='tight')
plt.show()