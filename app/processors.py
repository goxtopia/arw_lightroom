import numpy as np
import cv2
from scipy.interpolate import CubicSpline
import rawpy
from app.pipeline import BaseProcessor

class RawLoader:
    """
    Utility class to load and demosaic Sony ARW files with custom white balance multipliers.
    """
    @staticmethod
    def load(path: str, params: dict, half_size: bool = True) -> tuple[np.ndarray, list[float]]:
        with rawpy.imread(path) as raw:
            camera_wb = list(raw.camera_whitebalance)
            if not camera_wb or len(camera_wb) < 4 or all(v == 0.0 for v in camera_wb):
                # Fallback to standard daylight multipliers
                camera_wb = [2.0, 1.0, 1.5, 1.0]
            
            temp = float(params.get("temperature", 5500.0))
            tint = float(params.get("tint", 0.0))
            
            # Map temperature (Kelvin) and tint relative to default midpoint (5500K, Tint 0)
            f_temp = temp / 5500.0
            f_tint = 1.0 - (tint / 150.0)  # Positive tint reduces green (makes it magenta)
            
            r_mul = max(0.01, camera_wb[0] * f_temp)
            g0_mul = max(0.01, camera_wb[1] * f_tint)
            b_mul = max(0.01, camera_wb[2] / f_temp)
            g1_mul = max(0.01, camera_wb[3] * f_tint)
            
            user_wb = [r_mul, g0_mul, b_mul, g1_mul]
            
            if half_size:
                # Fast 8-bit demosaicing for preview
                img = raw.postprocess(
                    user_wb=user_wb,
                    half_size=True,
                    no_auto_bright=True,
                    output_bps=8
                )
                img_float = img.astype(np.float32)
                img_float /= 255.0
            else:
                # Full resolution 16-bit demosaicing for export
                img = raw.postprocess(
                    user_wb=user_wb,
                    half_size=False,
                    no_auto_bright=True,
                    output_bps=16
                )
                img_float = img.astype(np.float32)
                img_float /= 65535.0
                
            return img_float, camera_wb


class ExposureProcessor(BaseProcessor):
    def process(self, img: np.ndarray, params: dict) -> np.ndarray:
        ev = float(params.get("exposure", 0.0))
        if ev == 0.0:
            return img
            
        out = np.empty_like(img)
        h, w = img.shape[:2]
        total_pixels = h * w
        flat_img = img.reshape(-1, 3)
        flat_out = out.reshape(-1, 3)
        
        chunk_size = 5000000
        factor = np.float32(2.0 ** ev)
        
        for i in range(0, total_pixels, chunk_size):
            end_idx = min(i + chunk_size, total_pixels)
            np.clip(flat_img[i:end_idx] * factor, 0.0, 1.0, out=flat_out[i:end_idx])
            
        return out


class HighlightsShadowsProcessor(BaseProcessor):
    def process(self, img: np.ndarray, params: dict) -> np.ndarray:
        highlights = float(params.get("highlights", 0.0))  # Range: -1.0 to 1.0
        shadows = float(params.get("shadows", 0.0))        # Range: -1.0 to 1.0
        
        if highlights == 0.0 and shadows == 0.0:
            return img
            
        out = np.empty_like(img)
        h, w = img.shape[:2]
        total_pixels = h * w
        flat_img = img.reshape(-1, 3)
        flat_out = out.reshape(-1, 3)
        
        chunk_size = 5000000
        h_factor = np.float32(highlights * 0.5)
        s_factor = np.float32(shadows * 0.5)
        
        for i in range(0, total_pixels, chunk_size):
            end_idx = min(i + chunk_size, total_pixels)
            chunk = flat_img[i:end_idx]
            
            # Calculate luminance for chunk: shape (chunk_size,)
            luminance = 0.299 * chunk[:, 0] + 0.587 * chunk[:, 1] + 0.114 * chunk[:, 2]
            
            # Copy chunk to output
            chunk_out = chunk.copy()
            
            if highlights != 0.0:
                # Highlight mask: (luminance^2)
                h_mask = (luminance * luminance)[:, np.newaxis]
                h_mask *= h_factor
                h_mask += 1.0
                chunk_out *= h_mask
                
            if shadows != 0.0:
                # Shadow mask: (1 - luminance)^2
                diff = 1.0 - luminance
                s_mask = (diff * diff)[:, np.newaxis]
                if shadows > 0:
                    s_mask *= s_factor
                    one_minus_factor = 1.0 - s_mask
                    chunk_out *= one_minus_factor
                    chunk_out += s_mask
                else:
                    s_mask *= s_factor
                    s_mask += 1.0
                    chunk_out *= s_mask
                    
            np.clip(chunk_out, 0.0, 1.0, out=flat_out[i:end_idx])
            
        return out


class ContrastProcessor(BaseProcessor):
    def process(self, img: np.ndarray, params: dict) -> np.ndarray:
        contrast = float(params.get("contrast", 0.0))  # Range: -1.0 to 1.0
        if contrast == 0.0:
            return img
            
        out = np.empty_like(img)
        h, w = img.shape[:2]
        total_pixels = h * w
        flat_img = img.reshape(-1, 3)
        flat_out = out.reshape(-1, 3)
        
        chunk_size = 5000000
        c_factor = np.float32(contrast)
        
        for i in range(0, total_pixels, chunk_size):
            end_idx = min(i + chunk_size, total_pixels)
            chunk = flat_img[i:end_idx]
            
            diff = chunk - 0.5
            abs_diff = np.abs(diff)
            abs_diff *= -2.0
            abs_diff += 1.0
            abs_diff *= diff
            abs_diff *= c_factor
            abs_diff += chunk
            
            np.clip(abs_diff, 0.0, 1.0, out=flat_out[i:end_idx])
            
        return out


class SaturationProcessor(BaseProcessor):
    def process(self, img: np.ndarray, params: dict) -> np.ndarray:
        saturation = float(params.get("saturation", 0.0))  # Range: -1.0 to 1.0
        if saturation == 0.0:
            return img
            
        out = np.empty_like(img)
        h, w = img.shape[:2]
        total_pixels = h * w
        flat_img = img.reshape(-1, 3)
        flat_out = out.reshape(-1, 3)
        
        chunk_size = 5000000
        sat_factor = np.float32(1.0 + saturation)
        
        for i in range(0, total_pixels, chunk_size):
            end_idx = min(i + chunk_size, total_pixels)
            chunk = flat_img[i:end_idx]
            
            luminance = (0.299 * chunk[:, 0] + 0.587 * chunk[:, 1] + 0.114 * chunk[:, 2])[:, np.newaxis]
            chunk_out = chunk * sat_factor
            lum_factor = np.float32(1.0 - sat_factor)
            luminance *= lum_factor
            chunk_out += luminance
            
            np.clip(chunk_out, 0.0, 1.0, out=flat_out[i:end_idx])
            
        return out


class CurveProcessor(BaseProcessor):
    def _get_lut(self, points: list) -> np.ndarray:
        if not points or len(points) < 2:
            return None
            
        # Sort points by x coordinate
        points = sorted(points, key=lambda p: p[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        
        # Remove duplicate x coordinates to avoid CubicSpline singularity
        x_clean = [x[0]]
        y_clean = [y[0]]
        for i in range(1, len(x)):
            if x[i] != x[i-1]:
                x_clean.append(x[i])
                y_clean.append(y[i])
                
        if len(x_clean) < 2:
            return None
            
        try:
            if len(x_clean) == 2:
                lut = np.interp(np.linspace(0, 1, 256), x_clean, y_clean)
            else:
                cs = CubicSpline(x_clean, y_clean, bc_type='clamped')
                lut = cs(np.linspace(0, 1, 256))
        except Exception:
            # Fallback to linear interpolation in case spline fails
            lut = np.interp(np.linspace(0, 1, 256), x_clean, y_clean)
            
        return np.clip(lut, 0.0, 1.0).astype(np.float32)

    def _apply_curve_lut(self, img: np.ndarray, curve_lut: np.ndarray, channel_idx: int = None) -> np.ndarray:
        xp = np.linspace(0, 1, 256, dtype=np.float32)
        lut_fine = np.interp(np.linspace(0, 1, 4096, dtype=np.float32), xp, curve_lut).astype(np.float32)
        
        h, w = img.shape[:2]
        total_pixels = h * w
        chunk_size = 5000000
        
        if channel_idx is None:
            out = np.empty_like(img)
            flat_img = img.reshape(-1, 3)
            flat_out = out.reshape(-1, 3)
            
            for i in range(0, total_pixels, chunk_size):
                end_idx = min(i + chunk_size, total_pixels)
                chunk = flat_img[i:end_idx]
                indices = (chunk * 4095.0)
                np.clip(indices, 0.0, 4095.0, out=indices)
                flat_out[i:end_idx] = lut_fine[indices.astype(np.int32)]
            return out
        else:
            flat_chan = img[:, :, channel_idx].ravel()
            for i in range(0, total_pixels, chunk_size):
                end_idx = min(i + chunk_size, total_pixels)
                chunk = flat_chan[i:end_idx]
                indices = (chunk * 4095.0)
                np.clip(indices, 0.0, 4095.0, out=indices)
                flat_chan[i:end_idx] = lut_fine[indices.astype(np.int32)]
            return img

    def process(self, img: np.ndarray, params: dict) -> np.ndarray:
        curves = params.get("curves", {})
        if not curves:
            return img
            
        out = img
        
        # Apply combined RGB curve first
        rgb_pts = curves.get("rgb")
        rgb_lut = self._get_lut(rgb_pts)
        if rgb_lut is not None:
            out = self._apply_curve_lut(out, rgb_lut)
        else:
            # If no RGB curve, we copy the image so we don't modify the input in-place
            out = img.copy()
            
        # Apply channel-specific curves
        for idx, channel_name in enumerate(["red", "green", "blue"]):
            channel_pts = curves.get(channel_name)
            channel_lut = self._get_lut(channel_pts)
            if channel_lut is not None:
                out = self._apply_curve_lut(out, channel_lut, channel_idx=idx)
                
        return np.clip(out, 0.0, 1.0)


class LutProcessor(BaseProcessor):
    def __init__(self, lut_manager):
        self.lut_manager = lut_manager
        
    def process(self, img: np.ndarray, params: dict) -> np.ndarray:
        lut_name = params.get("lut")
        if not lut_name or lut_name == "None":
            return img
        return self.lut_manager.apply_lut(img, lut_name)


class NoiseReductionProcessor(BaseProcessor):
    def process(self, img: np.ndarray, params: dict) -> np.ndarray:
        strength = float(params.get("color_noise_reduction", 0.0))  # Range: 0.0 to 100.0
        if strength <= 0.0:
            return img
            
        # Convert to 8-bit RGB for OpenCV filtering using a memory-efficient chunked approach
        img_uint8 = np.empty(img.shape, dtype=np.uint8)
        h, w = img.shape[:2]
        total_pixels = h * w
        flat_img = img.reshape(-1, 3)
        flat_uint8 = img_uint8.reshape(-1, 3)
        chunk_size = 5000000
        
        for i in range(0, total_pixels, chunk_size):
            end_idx = min(i + chunk_size, total_pixels)
            flat_uint8[i:end_idx] = (flat_img[i:end_idx] * 255.0).clip(0, 255).astype(np.uint8)
            
        # Convert to YCrCb space
        ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Filter settings based on strength slider
        if strength < 30.0:
            d = 5
        elif strength < 70.0:
            d = 9
        else:
            d = 13
            
        sigma_color = 10.0 + strength * 1.4
        sigma_space = 3.0 + strength * 0.22
        
        try:
            # Attempt Joint Bilateral Filter with Y channel as structural guidance
            cr_filtered = cv2.ximgproc.jointBilateralFilter(y, cr, d, sigma_color, sigma_space)
            cb_filtered = cv2.ximgproc.jointBilateralFilter(y, cb, d, sigma_color, sigma_space)
            ycrcb_filtered = cv2.merge([y, cr_filtered, cb_filtered])
            rgb_filtered = cv2.cvtColor(ycrcb_filtered, cv2.COLOR_YCrCb2RGB)
            
            out = rgb_filtered.astype(np.float32)
            out /= 255.0
            return out
        except AttributeError:
            # Fallback to standard bilateral filtering on chroma channels if ximgproc is not compiled
            cr_filtered = cv2.bilateralFilter(cr, d, sigma_color, sigma_space)
            cb_filtered = cv2.bilateralFilter(cb, d, sigma_color, sigma_space)
            ycrcb_filtered = cv2.merge([y, cr_filtered, cb_filtered])
            rgb_filtered = cv2.cvtColor(ycrcb_filtered, cv2.COLOR_YCrCb2RGB)
            
            out = rgb_filtered.astype(np.float32)
            out /= 255.0
            return out
