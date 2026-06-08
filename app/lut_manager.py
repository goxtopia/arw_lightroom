import os
import numpy as np
from scipy.ndimage import map_coordinates

class LutManager:
    def __init__(self, luts_dir: str):
        self.luts_dir = luts_dir
        self.lut_cache = {}
        os.makedirs(self.luts_dir, exist_ok=True)
        self._generate_default_luts()

    def list_luts(self) -> list[str]:
        """Lists all available .cube files in the LUTs directory."""
        luts = [f for f in os.listdir(self.luts_dir) if f.endswith(".cube")]
        return ["None"] + sorted(luts)

    def load_lut(self, lut_name: str) -> np.ndarray:
        """Loads and parses a 3D LUT from a .cube file, caching it in memory."""
        if lut_name == "None":
            return None
            
        if lut_name in self.lut_cache:
            return self.lut_cache[lut_name]
            
        lut_path = os.path.join(self.luts_dir, lut_name)
        if not os.path.exists(lut_path):
            return None
            
        try:
            lut_data, size = self._parse_cube_file(lut_path)
            self.lut_cache[lut_name] = (lut_data, size)
            return lut_data, size
        except Exception as e:
            print(f"Error loading LUT {lut_name}: {e}")
            return None

    def _parse_cube_file(self, filepath: str) -> tuple[np.ndarray, int]:
        """Parses a standard Adobe .cube file format."""
        size = 0
        domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        values = []
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                    
                tokens = line.split()
                if tokens[0] == "LUT_3D_SIZE":
                    size = int(tokens[1])
                elif tokens[0] == "DOMAIN_MIN":
                    domain_min = np.array([float(x) for x in tokens[1:4]], dtype=np.float32)
                elif tokens[0] == "DOMAIN_MAX":
                    domain_max = np.array([float(x) for x in tokens[1:4]], dtype=np.float32)
                else:
                    try:
                        # Attempt to parse RGB values
                        rgb = [float(x) for x in tokens[:3]]
                        values.append(rgb)
                    except ValueError:
                        continue
                        
        if size == 0 or len(values) != size ** 3:
            raise ValueError(f"Invalid cube file: size {size} mismatch with data length {len(values)}")
            
        # Standard .cube has Red changing fastest, then Green, then Blue.
        # This maps directly to reshaping as (Blue, Green, Red, 3) where Blue is axis 0, Green is axis 1, Red is axis 2.
        lut_array = np.array(values, dtype=np.float32).reshape((size, size, size, 3))
        
        # Scale to match domain if needed (usually [0.0, 1.0])
        # For simplicity, we assume the LUT is scaled. If domain is different, we adjust it:
        if not np.allclose(domain_min, 0.0) or not np.allclose(domain_max, 1.0):
            lut_array = (lut_array - domain_min) / (domain_max - domain_min)
            
        return lut_array, size

    def apply_lut(self, img: np.ndarray, lut_name: str) -> np.ndarray:
        """
        Applies a 3D LUT to the image. Uses chunk-based vectorized trilinear interpolation
        via scipy.ndimage.map_coordinates to optimize speed and RAM.
        
        Args:
            img: Float32 image array normalized to [0.0, 1.0] with shape (H, W, 3).
            lut_name: Filename of the LUT to apply.
            
        Returns:
            The color-graded float32 image array.
        """
        lut_info = self.load_lut(lut_name)
        if lut_info is None:
            return img
            
        lut, N = lut_info
        
        h, w = img.shape[:2]
        total_pixels = h * w
        
        # Process in chunks of 1,000,000 pixels to optimize memory layout and CPU cache
        chunk_size = 1000000
        
        # Allocate output array
        out = np.empty_like(img)
        flat_img = img.reshape(-1, 3)
        flat_out = out.reshape(-1, 3)
        
        # Sequentially process slices
        for i in range(0, total_pixels, chunk_size):
            end_idx = min(i + chunk_size, total_pixels)
            chunk_img = flat_img[i:end_idx]
            
            # Map coordinates for mapping
            # Standard .cube has Red changing fastest (axis 2), then Green (axis 1), then Blue (axis 0)
            coords = np.vstack([
                chunk_img[:, 2] * (N - 1),  # Blue (Axis 0)
                chunk_img[:, 1] * (N - 1),  # Green (Axis 1)
                chunk_img[:, 0] * (N - 1)   # Red (Axis 2)
            ])
            
            for ch in range(3):
                flat_out[i:end_idx, ch] = map_coordinates(
                    lut[:, :, :, ch], coords, order=1, prefilter=False
                )
            
        return np.clip(out, 0.0, 1.0)

    def _generate_default_luts(self):
        """Generates sample LUTs so the user has immediate creative presets."""
        teal_orange_path = os.path.join(self.luts_dir, "Cinematic Teal & Orange.cube")
        if not os.path.exists(teal_orange_path):
            self._write_teal_orange_lut(teal_orange_path)
            
        warm_vintage_path = os.path.join(self.luts_dir, "Warm Vintage.cube")
        if not os.path.exists(warm_vintage_path):
            self._write_warm_vintage_lut(warm_vintage_path)

    def _write_teal_orange_lut(self, filepath: str):
        """Creates a gorgeous Hollywood-style Teal & Orange look."""
        N = 33
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Cinematic Teal & Orange LUT\n")
            f.write(f"LUT_3D_SIZE {N}\n\n")
            
            for b_idx in range(N):
                b = b_idx / (N - 1)
                for g_idx in range(N):
                    g = g_idx / (N - 1)
                    for r_idx in range(N):
                        r = r_idx / (N - 1)
                        
                        # Calculate luminance (standard BT.709)
                        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
                        
                        # Orange vector for highlights (warmer reds/yellows)
                        orange = np.array([y * 1.1 + 0.08, y * 0.95 + 0.02, y * 0.75 - 0.02])
                        # Teal vector for shadows (cooler cyan/blue)
                        teal = np.array([y * 0.75 - 0.02, y * 0.98 + 0.01, y * 1.08 + 0.06])
                        
                        # Blend based on luminance (soft transition)
                        t = np.clip((y - 0.1) / 0.8, 0.0, 1.0) # Highlight weight
                        blended = t * orange + (1.0 - t) * teal
                        
                        # Smoothly blend with the original color
                        final = 0.65 * np.array([r, g, b]) + 0.35 * blended
                        final = np.clip(final, 0.0, 1.0)
                        
                        f.write(f"{final[0]:.6f} {final[1]:.6f} {final[2]:.6f}\n")

    def _write_warm_vintage_lut(self, filepath: str):
        """Creates a nostalgic Warm Vintage warm-toned, low-contrast matte look."""
        N = 33
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Warm Vintage LUT\n")
            f.write(f"LUT_3D_SIZE {N}\n\n")
            
            for b_idx in range(N):
                b = b_idx / (N - 1)
                for g_idx in range(N):
                    g = g_idx / (N - 1)
                    for r_idx in range(N):
                        r = r_idx / (N - 1)
                        
                        # Lift shadows (faded/matte look)
                        r_lifted = 0.05 + 0.90 * r
                        g_lifted = 0.04 + 0.91 * g
                        b_lifted = 0.07 + 0.88 * b
                        
                        # Apply a warm tone curve shift
                        r_final = r_lifted ** 0.92  # Boost red midtones
                        g_final = g_lifted ** 0.97  # Slight boost
                        b_final = b_lifted ** 1.05  # Cool down shadows slightly
                        
                        # Warm color overlay
                        warm_color = np.array([r_final * 1.03, g_final * 1.01, b_final * 0.95])
                        final = np.clip(warm_color, 0.0, 1.0)
                        
                        f.write(f"{final[0]:.6f} {final[1]:.6f} {final[2]:.6f}\n")
