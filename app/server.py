import os
import threading
import urllib.parse
import queue
from typing import Dict, List
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pipeline import ImagePipeline
from app.processors import (
    RawLoader, ExposureProcessor, HighlightsShadowsProcessor,
    ContrastProcessor, SaturationProcessor, CurveProcessor,
    LutProcessor, NoiseReductionProcessor
)
from app.lut_manager import LutManager
from app.preset_manager import PresetManager

# App configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LUTS_DIR = os.path.join(BASE_DIR, "luts")
PRESETS_DIR = os.path.join(BASE_DIR, "presets")

# Managers
lut_manager = LutManager(LUTS_DIR)
preset_manager = PresetManager(PRESETS_DIR)

# FastAPI App
app = FastAPI(title="Sony ARW Photo Editor")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for RAW base images (demosaiced at preview resolution)
# Structure: { filepath: { "wb": (temp, tint), "img": np.ndarray, "camera_wb": list } }
raw_preview_cache: Dict[str, dict] = {}
cache_lock = threading.Lock()
MAX_CACHE_SIZE = 5

# Global background export state
export_status = {
    "running": False,
    "total": 0,
    "current": 0,
    "current_file": "",
    "errors": [],
    "completed": []
}

# Silent background export status tracking
silent_export_status = {
    "active_file": None,
    "completed": 0,
    "total_queued": 0
}

# Image Pipeline Setup
pipeline = ImagePipeline([
    ExposureProcessor(),
    HighlightsShadowsProcessor(),
    ContrastProcessor(),
    SaturationProcessor(),
    CurveProcessor(),
    LutProcessor(lut_manager),
    NoiseReductionProcessor()
])

# Models
class ScanRequest(BaseModel):
    folder_path: str

class PresetSaveRequest(BaseModel):
    name: str
    params: dict

class ExportRequest(BaseModel):
    files: List[str]
    params_map: Dict[str, dict]  # Maps file path to parameters dict

class ExportSingleRequest(BaseModel):
    filepath: str
    params: dict

# Utility to manage preview cache
def get_cached_raw_base(filepath: str, temp: float, tint: float) -> tuple[np.ndarray, list[float]]:
    global raw_preview_cache
    with cache_lock:
        # Check if cache hit with matching WB parameters
        if filepath in raw_preview_cache:
            cache_entry = raw_preview_cache[filepath]
            if cache_entry["wb"] == (temp, tint):
                # Move to end to preserve LRU order
                raw_preview_cache.pop(filepath)
                raw_preview_cache[filepath] = cache_entry
                return cache_entry["img"], cache_entry["camera_wb"]
        
        # Cache miss or WB changed: Load RAW base
        try:
            params = {"temperature": temp, "tint": tint}
            img_float, camera_wb = RawLoader.load(filepath, params, half_size=True)
            
            # Downsample to a standard preview width (e.g., max 1280px) for speed
            h, w = img_float.shape[:2]
            max_preview_dim = 1280
            if max(h, w) > max_preview_dim:
                scale = max_preview_dim / max(h, w)
                w_new = int(w * scale)
                h_new = int(h * scale)
                # Area interpolation is best for downscaling
                img_float = cv2.resize(img_float, (w_new, h_new), interpolation=cv2.INTER_AREA)
            
            # Cache the entry
            cache_entry = {
                "wb": (temp, tint),
                "img": img_float,
                "camera_wb": camera_wb
            }
            
            # Evict oldest if cache exceeded
            if len(raw_preview_cache) >= MAX_CACHE_SIZE:
                oldest = next(iter(raw_preview_cache))
                raw_preview_cache.pop(oldest)
                
            raw_preview_cache[filepath] = cache_entry
            return img_float, camera_wb
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load RAW image: {str(e)}")

# Background export worker
def batch_export_worker(files: List[str], params_map: Dict[str, dict]):
    global export_status
    export_status["running"] = True
    export_status["total"] = len(files)
    export_status["current"] = 0
    export_status["current_file"] = ""
    export_status["errors"] = []
    export_status["completed"] = []
    
    import gc
    
    # Define dedicated export pipeline
    export_pipeline = ImagePipeline([
        ExposureProcessor(),
        HighlightsShadowsProcessor(),
        ContrastProcessor(),
        SaturationProcessor(),
        CurveProcessor(),
        LutProcessor(lut_manager),
        NoiseReductionProcessor()
    ])
    
    for idx, filepath in enumerate(files):
        if not export_status["running"]:
            break
            
        filename = os.path.basename(filepath)
        export_status["current_file"] = filename
        
        try:
            # Copy parameters to avoid mutating the shared default dict in-place
            params = dict(params_map.get(filepath, params_map.get("default", {})))
            
            # If Auto NR is checked, calculate strength dynamically for this image's ISO rating
            if params.get("auto_nr", False):
                exif_meta = get_exif_metadata(filepath)
                iso = exif_meta.get("iso")
                if iso is not None:
                    try:
                        iso_val = float(iso)
                        if iso_val > 100.0:
                            import math
                            strength = 13.0 * math.log2(iso_val / 100.0)
                            params["color_noise_reduction"] = max(0.0, min(100.0, round(strength)))
                        else:
                            params["color_noise_reduction"] = 0.0
                    except Exception:
                        params["color_noise_reduction"] = 0.0
                else:
                    params["color_noise_reduction"] = 0.0
            
            # Load RAW at full 16-bit resolution
            img_float, _ = RawLoader.load(filepath, params, half_size=False)
            
            # Run image pipeline
            processed_img = export_pipeline.run(img_float, params)
            
            # Convert to 8-bit BGR for saving using a memory-efficient chunked approach
            img_uint8 = np.empty(processed_img.shape, dtype=np.uint8)
            h, w = processed_img.shape[:2]
            total_pixels = h * w
            flat_proc = processed_img.reshape(-1, 3)
            flat_uint8 = img_uint8.reshape(-1, 3)
            chunk_size = 5000000
            for i in range(0, total_pixels, chunk_size):
                end_idx = min(i + chunk_size, total_pixels)
                flat_uint8[i:end_idx] = (flat_proc[i:end_idx] * 255.0).clip(0, 255).astype(np.uint8)
                
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            # Save in an 'exports' subfolder within the same source directory
            source_dir = os.path.dirname(filepath)
            export_dir = os.path.join(source_dir, "exports")
            os.makedirs(export_dir, exist_ok=True)
            
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(export_dir, f"{base_name}_processed.png")
            
            cv2.imwrite(out_path, img_bgr)
            
            # Clean up references immediately to release memory
            del img_float
            del processed_img
            del img_uint8
            del img_bgr
            
            export_status["completed"].append(filename)
            export_status["current"] = idx + 1
        except Exception as e:
            err = f"Error processing {filename}: {str(e)}"
            export_status["errors"].append(err)
            export_status["current"] = idx + 1
            print(err)
        finally:
            # Force garbage collection to free large raw image buffers
            gc.collect()
            
    export_status["running"] = False

# Silent background queue for single image exports
silent_export_queue = queue.Queue()

def silent_export_worker():
    import gc
    # Dedicated export pipeline
    export_pipeline = ImagePipeline([
        ExposureProcessor(),
        HighlightsShadowsProcessor(),
        ContrastProcessor(),
        SaturationProcessor(),
        CurveProcessor(),
        LutProcessor(lut_manager),
        NoiseReductionProcessor()
    ])
    
    while True:
        try:
            item = silent_export_queue.get()
            if item is None:
                break
            filepath, params = item
            filename = os.path.basename(filepath)
            
            # Update background status
            silent_export_status["active_file"] = filename
            silent_export_status["total_queued"] = silent_export_queue.qsize() + 1
            
            # Load RAW at full 16-bit resolution
            img_float, _ = RawLoader.load(filepath, params, half_size=False)
            
            # Run image pipeline
            processed_img = export_pipeline.run(img_float, params)
            
            # Convert to 8-bit BGR for saving using a memory-efficient chunked approach
            img_uint8 = np.empty(processed_img.shape, dtype=np.uint8)
            h, w = processed_img.shape[:2]
            total_pixels = h * w
            flat_proc = processed_img.reshape(-1, 3)
            flat_uint8 = img_uint8.reshape(-1, 3)
            chunk_size = 5000000
            for i in range(0, total_pixels, chunk_size):
                end_idx = min(i + chunk_size, total_pixels)
                flat_uint8[i:end_idx] = (flat_proc[i:end_idx] * 255.0).clip(0, 255).astype(np.uint8)
                
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            # Save in an 'exports' subfolder within the same source directory
            source_dir = os.path.dirname(filepath)
            export_dir = os.path.join(source_dir, "exports")
            os.makedirs(export_dir, exist_ok=True)
            
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(export_dir, f"{base_name}_processed.png")
            
            cv2.imwrite(out_path, img_bgr)
            
            # Clean up references immediately to release memory
            del img_float
            del processed_img
            del img_uint8
            del img_bgr
            print(f"[SILENT EXPORT SUCCESS] Exported {filename}")
        except Exception as e:
            print(f"[SILENT EXPORT ERROR] Failed to export: {str(e)}")
        finally:
            silent_export_status["active_file"] = None
            silent_export_status["completed"] += 1
            silent_export_status["total_queued"] = silent_export_queue.qsize()
            gc.collect()
            silent_export_queue.task_done()

# Start silent export daemon thread
t = threading.Thread(target=silent_export_worker, daemon=True)
t.start()

# API Routes
@app.post("/api/scan_folder")
def scan_folder(req: ScanRequest):
    folder = req.folder_path
    if not os.path.exists(folder) or not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail="Invalid folder path")
        
    try:
        files = []
        for f in os.listdir(folder):
            if f.lower().endswith(".arw"):
                filepath = os.path.join(folder, f)
                size_bytes = os.path.getsize(filepath)
                size_mb = round(size_bytes / (1024 * 1024), 2)
                files.append({
                    "name": f,
                    "path": filepath,
                    "size": f"{size_mb} MB"
                })
        # Sort by filename
        files = sorted(files, key=lambda x: x["name"])
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/thumbnail")
def get_thumbnail(path: str = Query(...)):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        import rawpy
        from PIL import Image
        import io
        
        with rawpy.imread(path) as raw:
            try:
                # 1. Try to extract embedded JPEG preview
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    return Response(content=thumb.data, media_type="image/jpeg")
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    img = Image.fromarray(thumb.data)
                    img.thumbnail((320, 240))
                    out = io.BytesIO()
                    img.save(out, format="JPEG", quality=80)
                    return Response(content=out.getvalue(), media_type="image/jpeg")
            except Exception:
                pass
                
            # 2. Fallback to quick downsampled demosaic
            img = raw.postprocess(half_size=True, no_auto_bright=True, output_bps=8)
            # Resize using PIL for convenience
            pil_img = Image.fromarray(img)
            pil_img.thumbnail((320, 240))
            out = io.BytesIO()
            pil_img.save(out, format="JPEG", quality=75)
            return Response(content=out.getvalue(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate thumbnail: {str(e)}")

def get_exif_metadata(path: str) -> dict:
    metadata = {
        "make": "Sony",
        "model": "Alpha Camera",
        "iso": None,
        "shutter": None,
        "aperture": None,
        "focal_length": None
    }
    try:
        import rawpy
        from PIL import Image
        import io
        with rawpy.imread(path) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    img = Image.open(io.BytesIO(thumb.data))
                    exif = img.getexif()
                    
                    # Make & Model from Main IFD
                    metadata["make"] = str(exif.get(271, "Sony")).strip()
                    metadata["model"] = str(exif.get(272, "Alpha Camera")).strip()
                    
                    # Sub-IFD EXIF tags
                    sub_exif = exif.get_ifd(34665)
                    if sub_exif:
                        # ISO (Tag 34855)
                        metadata["iso"] = sub_exif.get(34855)
                        
                        # ExposureTime (Tag 33434)
                        shutter_val = sub_exif.get(33434)
                        if shutter_val:
                            shutter_float = float(shutter_val)
                            if shutter_float >= 1.0:
                                metadata["shutter"] = f"{shutter_float:.1f}s"
                            else:
                                try:
                                    metadata["shutter"] = f"1/{int(round(1.0 / shutter_float))}s"
                                except Exception:
                                    metadata["shutter"] = f"{shutter_val}s"
                                    
                        # FNumber (Tag 33437)
                        aperture_val = sub_exif.get(33437)
                        if aperture_val:
                            metadata["aperture"] = f"f/{float(aperture_val):.1f}"
                            
                        # FocalLength (Tag 37386)
                        focal_val = sub_exif.get(37386)
                        if focal_val:
                            metadata["focal_length"] = f"{float(focal_val):.0f}mm"
            except Exception:
                pass
    except Exception:
        pass
    return metadata

@app.get("/api/metadata")
def get_metadata(path: str = Query(...)):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        import rawpy
        with rawpy.imread(path) as raw:
            sizes = raw.sizes
            camera_wb = list(raw.camera_whitebalance)
            
        exif_meta = get_exif_metadata(path)
        
        return {
            "width": sizes.width,
            "height": sizes.height,
            "raw_width": sizes.raw_width,
            "raw_height": sizes.raw_height,
            "camera_wb": camera_wb,
            "camera_make": exif_meta["make"],
            "camera_model": exif_meta["model"],
            "iso": exif_meta["iso"],
            "shutter": exif_meta["shutter"],
            "aperture": exif_meta["aperture"],
            "focal_length": exif_meta["focal_length"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preview")
def get_preview(params: dict):
    filepath = params.get("path")
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File path not found")
        
    temp = float(params.get("temperature", 5500.0))
    tint = float(params.get("tint", 0.0))
    
    # 1. Fetch from cache or demosaic RAW base
    img_base, camera_wb = get_cached_raw_base(filepath, temp, tint)
    
    # 2. Run remaining downstream processors
    processed_img = pipeline.run(img_base, params)
    
    # 3. Convert back to 8-bit BGR for JPEG encoding
    img_uint8 = (processed_img * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    # 4. Encode to JPEG
    success, encoded_img = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise HTTPException(status_code=500, detail="JPEG compression failed")
        
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.get("/api/presets")
def list_presets():
    return {"presets": preset_manager.list_presets()}

@app.get("/api/presets/{name}")
def get_preset(name: str):
    preset = preset_manager.load_preset(name)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return preset

@app.post("/api/presets")
def save_preset(req: PresetSaveRequest):
    success = preset_manager.save_preset(req.name, req.params)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save preset")
    return {"status": "success", "presets": preset_manager.list_presets()}

@app.get("/api/luts")
def list_luts():
    return {"luts": lut_manager.list_luts()}

@app.post("/api/export")
def export_files(req: ExportRequest, background_tasks: BackgroundTasks):
    global export_status
    if export_status["running"]:
        raise HTTPException(status_code=400, detail="An export batch is already running")
        
    background_tasks.add_task(batch_export_worker, req.files, req.params_map)
    return {"status": "started", "total": len(req.files)}

@app.get("/api/export_status")
def get_export_status():
    global export_status
    return export_status

@app.post("/api/export_cancel")
def cancel_export():
    global export_status
    if export_status["running"]:
        export_status["running"] = False
        return {"status": "cancelled"}
    return {"status": "not running"}

@app.post("/api/export_single")
def export_single(req: ExportSingleRequest):
    if not os.path.exists(req.filepath):
        raise HTTPException(status_code=404, detail="File not found")
        
    params = dict(req.params)
    # If Auto NR is checked, calculate strength dynamically for this image's ISO rating
    if params.get("auto_nr", False):
        exif_meta = get_exif_metadata(req.filepath)
        iso = exif_meta.get("iso")
        if iso is not None:
            try:
                iso_val = float(iso)
                if iso_val > 100.0:
                    import math
                    strength = 13.0 * math.log2(iso_val / 100.0)
                    params["color_noise_reduction"] = max(0.0, min(100.0, round(strength)))
                else:
                    params["color_noise_reduction"] = 0.0
            except Exception:
                params["color_noise_reduction"] = 0.0
        else:
            params["color_noise_reduction"] = 0.0
            
    # Increment total queued counter
    silent_export_status["total_queued"] = silent_export_queue.qsize() + 1
    silent_export_queue.put((req.filepath, params))
    return {"status": "queued"}

@app.get("/api/silent_export_status")
def get_silent_export_status():
    global silent_export_status, silent_export_queue
    return {
        "active_file": silent_export_status["active_file"],
        "queue_size": silent_export_queue.qsize(),
        "total_queued": silent_export_status["total_queued"]
    }

# Serve Frontend static assets
static_path = os.path.join(BASE_DIR, "static")
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
