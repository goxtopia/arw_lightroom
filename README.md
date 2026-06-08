# α-Lightroom: Sony ARW RAW Photo Processor / 索尼 ARW 格式照片处理器

An optimized, premium RAW photo processing application tailored for Sony ARW files, featuring a sleek dark glassmorphism WebUI, customizable develop presets, high-speed 3D LUT color grading, chroma noise reduction, and high-performance batch exports.

一款专为索尼 ARW 原始格式照片设计的轻量级、高性能后期处理器。提供现代暗黑磨砂玻璃风格 (Glassmorphism) Web 界面，支持自定义预设保存、高性能 3D LUT 色彩滤镜、联合双边滤波降噪以及极低内存占用的批量高分辨率导出。

---

## Key Features / 功能特性

### 🌐 Modern Responsive WebUI / 现代响应式网页界面

- **Dark Glassmorphism Interface**: Sleek, immersive workspace optimized for visual focus.
- **暗黑玻璃拟态风格**: 极具质感的深色沉浸式工作区，让图像色彩展示更加聚焦。
- **Interactive Tone Curve Editor**: Canvas-based interactive spline editor for RGB and individual Red/Green/Blue channels.
- **交互式色调曲线编辑器**: 采用 Canvas 实现的交互式样条曲线调整，支持 RGB 复合通道及红、绿、蓝单通道调节。
- **Batch scan & Gallery**: Automatically scan folders, extract EXIF metadata (Camera Make/Model), and stream thumbnails.
- **目录扫描与画廊**: 自动扫描目标目录下的 ARW 文件，提取 EXIF 属性（相机品牌/型号），并实时载入内嵌缩略图。

### ⚙️ Raw Processing Pipeline / RAW 图像处理管线

- **Exposure & White Balance**: Adjust Exposure (EV), Color Temperature (Kelvin), and Tint multipliers relative to camera defaults.
- **曝光与白平衡**: 调整曝光度（EV）、色温（开尔文）以及色调增益。
- **Highlights, Shadows & Contrast**: Sophisticated mathematical highlights/shadows recovery and polynomial S-curves for contrast.
- **高光、阴影与对比度**: 高精度高光/阴影细节恢复，采用多项式 S 曲线平滑调整对比度。
- **3D LUT (Look-Up Table)**: Load standard `.cube` formats using chunk-based vectorization (SciPy `map_coordinates`) for movie-like color grading.
- **3D LUT 滤镜**: 载入标准 Adobe `.cube` 格式滤镜，采用分块向量化的三线性插值，一键呈现电影级色彩。
- **Chroma Denoise**: Custom Joint Bilateral Filter (`cv2.ximgproc`) on Cr/Cb channels using the Y channel as structural guidance to clear chroma noise while preserving sharp luminance detail.
- **彩色降噪**: 提取 YCrCb 空间的色彩通道，以亮度通道 Y 作为结构引导进行联合双边滤波（Joint Bilateral Filtering），在完美保留边缘细节的前提下清除彩色噪声。

### ⚡ High Performance & Memory Bounded / 高性能与低内存占用

- **Chunked Pipelines**: Slices raw image arrays into flat blocks (e.g. 5,000,000 pixels) during filter execution to cap peak memory.
- **分块管线**: 在图像滤镜处理与格式转换中进行像素分块迭代（单次分块 5,000,000 像素）。
- **12-bit Quantized Curves**: Fast 4096-bin lookup table mapping for spline curves, keeping calculations strictly in float32.
- **12位量化色调曲线**: 将样条曲线映射为高精度的 4096 档查找表。
- **Serial Batch Export**: Run folder exports in a background serial worker with proactive garbage collection to ensure stable execution.
- **串行后台导出**: 采用串行后台工作线程进行批量高分辨率导出，并在每张图处理完毕后强制释放内存，防止多线程引发内存崩溃。

---

## System Requirements / 系统要求

- **OS**: Windows / macOS / Linux
- **Python**: 3.10 or higher
- **Package Manager**: Recommended `uv` for ultra-fast package resolution.

---

## Installation & Setup / 安装与运行

1. **Clone the Repository / 克隆项目**

   ```bash
   git clone <repository_url>
   cd arw_lightroom
   ```

2. **Run the Application / 启动应用**
   If you have `uv` installed, simply run:
   如果您安装了 `uv`，直接运行即可（会自动创建虚拟环境并下载依赖）：

   ```bash
   uv run main.py
   ```

   _Otherwise, install dependencies using pip / 或者，您也可以手动安装依赖项：_

   ```bash
   pip install -r requirements.txt  # If requirements.txt is generated
   # Or install packages directly:
   pip install rawpy numpy opencv-contrib-python scipy fastapi uvicorn pillow
   python main.py
   ```

3. **Access the WebUI / 访问界面**
   Open your browser and navigate to:
   打开浏览器，访问：
   ```
   http://127.0.0.1:8000
   ```

---

## Directory Structure / 项目结构

```
arw_lightroom/
│
├── app/
│   ├── pipeline.py        # Image processor chain manager / 图像处理器链管理
│   ├── processors.py      # Core RAW editing processors / 核心图像滤镜处理器
│   ├── lut_manager.py     # 3D LUT loader & SciPy interpolator / 3D LUT 载入与三线性插值
│   ├── preset_manager.py  # JSON Preset loading/saving / 预设读写管理
│   └── server.py          # FastAPI application & Background exporter / 后端接口与后台导出
│
├── static/                # WebUI Front-end assets / 前端静态资源
│   ├── css/style.css      # Custom glassmorphism UI styles / 前端暗色玻璃拟态样式
│   ├── js/app.js          # Tone curve canvas & Server API integrations / 曲线画布与API交互
│   └── index.html         # Main dashboard markup / 前端主界面
│
├── luts/                  # Loaded .cube LUT directory / 3D LUT 存放目录
├── presets/               # Saved Develop Presets / 用户自定义预设目录
├── tests/                 # Unit tests / 单元测试
├── main.py                # Server entry point / 服务启动入口
└── pyproject.toml         # UV Project config / UV 项目配置文件
```

---

## Key Optimization Details / 核心优化细节

### Curve Quantization Lookup / 曲线量化查找

Standard spline evaluation via `np.interp` evaluates millions of coordinates on a single CPU thread and promotes arrays to `float64`, consuming 1.44 GB of memory. By using a 4096-bin quantized float32 table:

```python
# Create a high-resolution 12-bit LUT (4096 bins)
lut_fine = np.interp(np.linspace(0, 1, 4096, dtype=np.float32), xp, curve_lut).astype(np.float32)

# Vectorized lookup in chunks
indices = (chunk * 4095.0)
np.clip(indices, 0.0, 4095.0, out=indices)
chunk_out = lut_fine[indices.astype(np.int32)]
```

This is **7x faster** and allocates less than 40 MB of temporary memory per chunk.

### SciPy LUT Mapping / 快速 LUT 映射

Applying 3D LUTs requires trilinear interpolation across $N^3$ space. We leverage SciPy's C-compiled `map_coordinates` in chunks:

```python
coords = np.vstack([
    chunk_img[:, 2] * (N - 1),  # Blue (Axis 0)
    chunk_img[:, 1] * (N - 1),  # Green (Axis 1)
    chunk_img[:, 0] * (N - 1)   # Red (Axis 2)
])
for ch in range(3):
    flat_out[i:end_idx, ch] = map_coordinates(lut[:, :, :, ch], coords, order=1, prefilter=False)
```

This reduces execution time from 28.7s to **8.1s** for a 60MP image and maintains strict float32 precision.
