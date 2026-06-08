/**
 * α-Lightroom - Frontend Application Logic
 */

// Application State
const state = {
    folderPath: "",
    files: [],
    selectedFiles: new Set(), // Set of file paths selected for export
    activeFilePath: null,
    activeFilename: null,
    cameraWb: [1.0, 1.0, 1.0, 1.0], // Default multiplier fallback
    presets: [],
    luts: [],
    exportStatusTimer: null,
    
    // Develop parameters for the current file
    params: {
        exposure: 0.0,
        contrast: 0.0,
        highlights: 0.0,
        shadows: 0.0,
        saturation: 0.0,
        temperature: 5500.0,
        tint: 0.0,
        color_noise_reduction: 0.0,
        lut: "None",
        curves: {
            rgb: [[0.0, 0.0], [1.0, 1.0]],
            red: [[0.0, 0.0], [1.0, 1.0]],
            green: [[0.0, 0.0], [1.0, 1.0]],
            blue: [[0.0, 0.0], [1.0, 1.0]]
        }
    },
    
    // Spline curve editor variables
    curve: {
        channel: "rgb",
        selectedPointIdx: -1,
        dragged: false,
        padding: 15,
        gridSize: 220
    }
};

// Default parameters for reset
const DEFAULT_PARAMS = {
    exposure: 0.0,
    contrast: 0.0,
    highlights: 0.0,
    shadows: 0.0,
    saturation: 0.0,
    temperature: 5500.0,
    tint: 0.0,
    color_noise_reduction: 0.0,
    lut: "None",
    curves: {
        rgb: [[0.0, 0.0], [1.0, 1.0]],
        red: [[0.0, 0.0], [1.0, 1.0]],
        green: [[0.0, 0.0], [1.0, 1.0]],
        blue: [[0.0, 0.0], [1.0, 1.0]]
    }
};

// UI Elements
const el = {
    folderPath: document.getElementById("folder-path"),
    btnScan: document.getElementById("btn-scan"),
    statusText: document.getElementById("status-text"),
    presetsList: document.getElementById("presets-list"),
    fileCount: document.getElementById("file-count"),
    btnSelectAll: document.getElementById("btn-select-all"),
    btnDeselectAll: document.getElementById("btn-deselect-all"),
    searchFiles: document.getElementById("search-files"),
    fileGrid: document.getElementById("file-grid"),
    currentFilename: document.getElementById("current-filename"),
    exifDetails: document.getElementById("exif-details"),
    btnBeforeAfter: document.getElementById("btn-before-after"),
    previewImage: document.getElementById("preview-image"),
    imageLoader: document.getElementById("image-loader"),
    viewerPlaceholder: document.getElementById("viewer-placeholder"),
    curveChannel: document.getElementById("curve-channel"),
    btnResetCurve: document.getElementById("btn-reset-curve"),
    curveCanvas: document.getElementById("curve-canvas"),
    
    // Sliders
    slideTemp: document.getElementById("slide-temp"),
    valTemp: document.getElementById("val-temp"),
    slideTint: document.getElementById("slide-tint"),
    valTint: document.getElementById("val-tint"),
    btnAsShotWb: document.getElementById("btn-as-shot-wb"),
    
    slideExposure: document.getElementById("slide-exposure"),
    valExposure: document.getElementById("val-exposure"),
    slideContrast: document.getElementById("slide-contrast"),
    valContrast: document.getElementById("val-contrast"),
    slideSaturation: document.getElementById("slide-saturation"),
    valSaturation: document.getElementById("val-saturation"),
    
    slideHighlights: document.getElementById("slide-highlights"),
    valHighlights: document.getElementById("val-highlights"),
    slideShadows: document.getElementById("slide-shadows"),
    valShadows: document.getElementById("val-shadows"),
    
    slideChromaNr: document.getElementById("slide-chroma-nr"),
    valChromaNr: document.getElementById("val-chroma-nr"),
    
    lutSelect: document.getElementById("lut-select"),
    newPresetName: document.getElementById("new-preset-name"),
    btnSavePreset: document.getElementById("btn-save-preset"),
    selectedExportCount: document.getElementById("selected-export-count"),
    btnExportBatch: document.getElementById("btn-export-batch"),
    
    // Filmstrip
    filmstripCarousel: document.getElementById("filmstrip-carousel"),
    
    // Modal
    exportModal: document.getElementById("export-modal"),
    exportProgressFill: document.getElementById("export-progress-fill"),
    exportProgressText: document.getElementById("export-progress-text"),
    exportProgressPercentage: document.getElementById("export-progress-percentage"),
    exportCurrentFileName: document.getElementById("export-current-file-name"),
    exportLogOutput: document.getElementById("export-log-output"),
    btnCancelExport: document.getElementById("btn-cancel-export"),
    btnCloseExport: document.getElementById("btn-close-export"),
    btnCancelExportModal: document.getElementById("btn-cancel-export-modal")
};

// Canvas context
const ctx = el.curveCanvas.getContext("2d");

// Throttle/Debounce Variables for real-time sliders
let previewTimeout = null;
let activeRequest = null;

// Initialize Page
document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
    loadPresets();
    loadLuts();
    drawCurve();
});

// Event Listeners Configuration
function initEventListeners() {
    el.btnScan.addEventListener("click", scanFolder);
    el.folderPath.addEventListener("keyup", (e) => {
        if (e.key === "Enter") scanFolder();
    });
    
    el.btnSelectAll.addEventListener("click", selectAllFiles);
    el.btnDeselectAll.addEventListener("click", deselectAllFiles);
    el.searchFiles.addEventListener("input", filterFiles);
    
    // Sliders input events (real-time label updates)
    setupSlider(el.slideExposure, el.valExposure, "", 2, true);
    setupSlider(el.slideContrast, el.valContrast, "", 2, true);
    setupSlider(el.slideSaturation, el.valSaturation, "", 2, true);
    setupSlider(el.slideHighlights, el.valHighlights, "", 2, true);
    setupSlider(el.slideShadows, el.valShadows, "", 2, true);
    setupSlider(el.slideTemp, el.valTemp, " K", 0, false);
    setupSlider(el.slideTint, el.valTint, "", 0, true);
    setupSlider(el.slideChromaNr, el.valChromaNr, "", 0, false);
    
    // WB Controls
    el.btnAsShotWb.addEventListener("click", () => {
        el.slideTemp.value = 5500;
        el.slideTint.value = 0;
        updateSliderLabel(el.slideTemp, el.valTemp, " K");
        updateSliderLabel(el.slideTint, el.valTint, "");
        state.params.temperature = 5500.0;
        state.params.tint = 0.0;
        triggerPreviewUpdate();
    });
    
    // LUT changed
    el.lutSelect.addEventListener("change", (e) => {
        state.params.lut = e.target.value;
        triggerPreviewUpdate();
    });
    
    // Before / After Comparison
    el.btnBeforeAfter.addEventListener("mousedown", showBeforeImage);
    el.btnBeforeAfter.addEventListener("mouseup", showAfterImage);
    el.btnBeforeAfter.addEventListener("mouseleave", showAfterImage);
    el.btnBeforeAfter.addEventListener("touchstart", showBeforeImage);
    el.btnBeforeAfter.addEventListener("touchend", showAfterImage);
    
    // Presets operations
    el.btnSavePreset.addEventListener("click", savePreset);
    
    // Curves adjustments
    el.curveChannel.addEventListener("change", (e) => {
        state.curve.channel = e.target.value;
        state.curve.selectedPointIdx = -1;
        drawCurve();
    });
    el.btnResetCurve.addEventListener("click", () => {
        state.params.curves[state.curve.channel] = [[0.0, 0.0], [1.0, 1.0]];
        state.curve.selectedPointIdx = -1;
        drawCurve();
        triggerPreviewUpdate();
    });
    
    // Curve canvas mouse/touch bindings
    el.curveCanvas.addEventListener("mousedown", onCurveMouseDown);
    el.curveCanvas.addEventListener("mousemove", onCurveMouseMove);
    window.addEventListener("mouseup", onCurveMouseUp);
    el.curveCanvas.addEventListener("contextmenu", onCurveRightClick);
    
    // Export Operations
    el.btnExportBatch.addEventListener("click", startBatchExport);
    el.btnCancelExport.addEventListener("click", cancelBatchExport);
    el.btnCancelExportModal.addEventListener("click", () => el.exportModal.style.display = "none");
    el.btnCloseExport.addEventListener("click", () => el.exportModal.style.display = "none");
}

// Slider Helper to map values instantly
function setupSlider(sliderEl, labelEl, suffix, decimals, showSign) {
    updateSliderLabel(sliderEl, labelEl, suffix, decimals, showSign);
    sliderEl.addEventListener("input", (e) => {
        updateSliderLabel(sliderEl, labelEl, suffix, decimals, showSign);
        let paramName = sliderEl.id.replace("slide-", "").replace("chroma-nr", "color_noise_reduction");
        if (paramName === "temp") paramName = "temperature";
        state.params[paramName] = parseFloat(e.target.value);
        triggerPreviewUpdate();
    });
}

function updateSliderLabel(sliderEl, labelEl, suffix, decimals = 0, showSign = false) {
    let val = parseFloat(sliderEl.value);
    let str = val.toFixed(decimals);
    if (showSign && val > 0) {
        str = "+" + str;
    }
    labelEl.textContent = str + suffix;
}

// Fetch APIs
async function scanFolder() {
    const path = el.folderPath.value.trim();
    if (!path) {
        alert("Please enter a valid directory path.");
        return;
    }
    
    updateStatus("Scanning folder...", "pulse");
    
    try {
        const res = await fetch("/api/scan_folder", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ folder_path: path })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Scanning failed");
        }
        
        const data = await res.json();
        state.folderPath = path;
        state.files = data.files;
        state.selectedFiles.clear();
        
        el.fileCount.textContent = `${state.files.length} files`;
        renderFileGrid();
        renderFilmstrip();
        updateExportCount();
        
        if (state.files.length > 0) {
            document.querySelector(".app-container").classList.add("has-filmstrip");
            updateStatus(`Found ${state.files.length} raw files`, "success");
            // Auto-load first file
            loadFile(state.files[0]);
        } else {
            document.querySelector(".app-container").classList.remove("has-filmstrip");
            updateStatus("No ARW files found", "warning");
            el.fileGrid.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="image-off" class="large-icon"></i>
                    <p>No Sony .ARW files found in this folder.</p>
                </div>
            `;
            lucide.createIcons();
        }
    } catch (e) {
        document.querySelector(".app-container").classList.remove("has-filmstrip");
        updateStatus("Scanning failed", "error");
        alert(`Error: ${e.message}`);
    }
}

function renderFileGrid() {
    el.fileGrid.innerHTML = "";
    
    if (state.files.length === 0) {
        return;
    }
    
    state.files.forEach(file => {
        const item = document.createElement("div");
        item.className = `file-grid-item ${state.activeFilePath === file.path ? 'active' : ''}`;
        item.dataset.path = file.path;
        
        // Checkbox for batch select
        const chk = document.createElement("input");
        chk.type = "checkbox";
        chk.checked = state.selectedFiles.has(file.path);
        chk.addEventListener("change", (e) => {
            e.stopPropagation();
            if (e.target.checked) {
                state.selectedFiles.add(file.path);
            } else {
                state.selectedFiles.delete(file.path);
            }
            updateExportCount();
            // Sync with carousel border
            syncFilmstripSelection(file.path, e.target.checked);
        });
        
        // Thumbnail
        const thumb = document.createElement("img");
        thumb.className = "file-thumb";
        thumb.src = `/api/thumbnail?path=${encodeURIComponent(file.path)}`;
        thumb.loading = "lazy";
        
        // Details
        const details = document.createElement("div");
        details.className = "file-details";
        
        const name = document.createElement("span");
        name.className = "file-name";
        name.textContent = file.name;
        
        const size = document.createElement("span");
        size.className = "file-size";
        size.textContent = file.size;
        
        details.appendChild(name);
        details.appendChild(size);
        
        item.appendChild(chk);
        item.appendChild(thumb);
        item.appendChild(details);
        
        item.addEventListener("click", () => {
            loadFile(file);
        });
        
        el.fileGrid.appendChild(item);
    });
}

function renderFilmstrip() {
    el.filmstripCarousel.innerHTML = "";
    state.files.forEach(file => {
        const item = document.createElement("div");
        item.className = `filmstrip-item ${state.activeFilePath === file.path ? 'active' : ''}`;
        item.dataset.path = file.path;
        
        const img = document.createElement("img");
        img.className = "filmstrip-thumb";
        img.src = `/api/thumbnail?path=${encodeURIComponent(file.path)}`;
        img.loading = "lazy";
        
        const label = document.createElement("span");
        label.className = "filmstrip-label";
        label.textContent = file.name;
        
        item.appendChild(img);
        item.appendChild(label);
        
        item.addEventListener("click", () => {
            loadFile(file);
        });
        
        el.filmstripCarousel.appendChild(item);
    });
}

function syncFilmstripSelection(filepath, isSelected) {
    // We could add visual checkmarks on filmstrip as well, but keeping it simple
}

function selectAllFiles() {
    state.files.forEach(f => state.selectedFiles.add(f.path));
    // Check all checkboxes in UI
    el.fileGrid.querySelectorAll("input[type='checkbox']").forEach(chk => chk.checked = true);
    updateExportCount();
}

function deselectAllFiles() {
    state.selectedFiles.clear();
    el.fileGrid.querySelectorAll("input[type='checkbox']").forEach(chk => chk.checked = false);
    updateExportCount();
}

function updateExportCount() {
    const cnt = state.selectedFiles.size;
    el.selectedExportCount.textContent = `${cnt} files selected for batch export`;
    el.btnExportBatch.disabled = cnt === 0;
}

function filterFiles(e) {
    const q = e.target.value.toLowerCase().trim();
    const items = el.fileGrid.querySelectorAll(".file-grid-item");
    items.forEach(item => {
        const name = item.querySelector(".file-name").textContent.toLowerCase();
        if (name.includes(q)) {
            item.style.display = "flex";
        } else {
            item.style.display = "none";
        }
    });
}

// Load active file
async function loadFile(file) {
    if (state.activeFilePath === file.path) return;
    
    state.activeFilePath = file.path;
    state.activeFilename = file.name;
    
    // Highlight items in grid & filmstrip
    el.fileGrid.querySelectorAll(".file-grid-item").forEach(item => {
        item.classList.toggle("active", item.dataset.path === file.path);
    });
    el.filmstripCarousel.querySelectorAll(".filmstrip-item").forEach(item => {
        item.classList.toggle("active", item.dataset.path === file.path);
    });
    
    // Show loading placeholders
    el.viewerPlaceholder.style.display = "none";
    el.previewImage.style.display = "none";
    el.imageLoader.style.display = "flex";
    el.currentFilename.textContent = file.name;
    el.exifDetails.textContent = "Loading EXIF...";
    
    // Reset parameters to defaults (or we can preserve parameters if user wants to copy-paste edit settings.
    // Preserving current slider settings allows user to apply adjustments across multiple images like Lightroom sync!)
    // Let's preserve current adjustments for convenience, which is exactly Lightroom's behavior when moving between files!
    
    try {
        // Get exif metadata
        const metadataRes = await fetch(`/api/metadata?path=${encodeURIComponent(file.path)}`);
        if (metadataRes.ok) {
            const meta = await metadataRes.json();
            state.cameraWb = meta.camera_wb;
            el.exifDetails.textContent = `${meta.camera_make} ${meta.camera_model} | ${meta.width}x${meta.height}`;
        }
    } catch (e) {
        el.exifDetails.textContent = "Sony RAW Image";
    }
    
    // Fetch and display preview
    refreshPreview();
}

// Debounced Preview Trigger
function triggerPreviewUpdate() {
    if (!state.activeFilePath) return;
    
    if (previewTimeout) {
        clearTimeout(previewTimeout);
    }
    
    // 120ms debounce for sliders dragging
    previewTimeout = setTimeout(refreshPreview, 120);
}

// Perform POST /api/preview
async function refreshPreview() {
    if (!state.activeFilePath) return;
    
    // Cancel active network request if it's still running (keeps sliders snappy!)
    if (activeRequest) {
        activeRequest.abort();
    }
    
    const controller = new AbortController();
    activeRequest = controller;
    
    // Build parameters body
    const params = {
        path: state.activeFilePath,
        exposure: state.params.exposure,
        contrast: state.params.contrast,
        highlights: state.params.highlights,
        shadows: state.params.shadows,
        saturation: state.params.saturation,
        temperature: state.params.temperature,
        tint: state.params.tint,
        color_noise_reduction: state.params.color_noise_reduction,
        lut: state.params.lut,
        curves: state.params.curves
    };
    
    try {
        const res = await fetch("/api/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
            signal: controller.signal
        });
        
        if (!res.ok) throw new Error("Preview failed");
        
        const blob = await res.blob();
        const imgUrl = URL.createObjectURL(blob);
        
        el.previewImage.src = imgUrl;
        el.previewImage.style.display = "block";
        el.imageLoader.style.display = "none";
        
        updateStatus("Demosaiced successfully", "success");
    } catch (e) {
        if (e.name !== "AbortError") {
            console.error("Preview render failed:", e);
            updateStatus("Preview render failed", "error");
        }
    } finally {
        if (activeRequest === controller) {
            activeRequest = null;
        }
    }
}

// Before/After comparison
let beforeImgUrl = null;
async function showBeforeImage() {
    if (!state.activeFilePath) return;
    
    el.imageLoader.style.display = "flex";
    
    // Fetch preview with default/zero adjustments
    const baseParams = Object.assign({}, DEFAULT_PARAMS, { path: state.activeFilePath });
    try {
        const res = await fetch("/api/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(baseParams)
        });
        if (res.ok) {
            const blob = await res.blob();
            beforeImgUrl = URL.createObjectURL(blob);
            el.previewImage.src = beforeImgUrl;
            updateStatus("Showing Original RAW", "warning");
        }
    } catch (e) {
        console.error(e);
    } finally {
        el.imageLoader.style.display = "none";
    }
}

function showAfterImage() {
    if (!state.activeFilePath) return;
    if (beforeImgUrl) {
        URL.revokeObjectURL(beforeImgUrl);
        beforeImgUrl = null;
    }
    refreshPreview();
}

// Curve Spline Canvas implementation
function drawCurve() {
    const size = state.curve.gridSize;
    const padding = state.curve.padding;
    const innerSize = size - 2 * padding;
    
    // Clear
    ctx.clearRect(0, 0, size, size);
    
    // Draw grid background
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    
    // Horizontal and vertical gridlines at 25%, 50%, 75%
    for (let i = 1; i <= 3; i++) {
        const ratio = i * 0.25;
        const pos = padding + ratio * innerSize;
        // Verticals
        ctx.beginPath();
        ctx.moveTo(pos, padding);
        ctx.lineTo(pos, size - padding);
        ctx.stroke();
        // Horizontals
        ctx.beginPath();
        ctx.moveTo(padding, pos);
        ctx.lineTo(size - padding, pos);
        ctx.stroke();
    }
    
    // Draw diagonal reference line (dashed)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padding, size - padding);
    ctx.lineTo(size - padding, padding);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Get points for active channel
    const pts = state.params.curves[state.curve.channel];
    
    // Interpolate points using spline
    const xCoords = pts.map(p => p[0]);
    const yCoords = pts.map(p => p[1]);
    const spline = createSpline(xCoords, yCoords);
    
    // Draw the curve line
    ctx.beginPath();
    ctx.strokeStyle = getChannelColor(state.curve.channel);
    ctx.lineWidth = 2.5;
    
    for (let px = 0; px <= innerSize; px++) {
        const x = px / innerSize;
        const y = spline(x);
        const cx = padding + px;
        const cy = size - padding - y * innerSize;
        
        if (px === 0) {
            ctx.moveTo(cx, cy);
        } else {
            ctx.lineTo(cx, cy);
        }
    }
    ctx.stroke();
    
    // Draw control points
    pts.forEach((p, idx) => {
        const cx = padding + p[0] * innerSize;
        const cy = size - padding - p[1] * innerSize;
        
        ctx.beginPath();
        ctx.arc(cx, cy, idx === state.curve.selectedPointIdx ? 6 : 4, 0, 2 * Math.PI);
        ctx.fillStyle = idx === state.curve.selectedPointIdx ? "#ffffff" : getChannelColor(state.curve.channel);
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#151518";
        ctx.stroke();
    });
}

function getChannelColor(chan) {
    if (chan === "red") return "#e74c3c";
    if (chan === "green") return "#2ecc71";
    if (chan === "blue") return "#3498db";
    return "#f39c12"; // RGB / Master Gold
}

// Spline Interpolation Logic (Monotone Cubic Spline)
function createSpline(x, y) {
    const n = x.length;
    if (n < 2) return (t) => t;
    
    const delta = [];
    const d = [];
    
    for (let i = 0; i < n - 1; i++) {
        delta.push((y[i+1] - y[i]) / (x[i+1] - x[i]));
    }
    
    d[0] = delta[0];
    for (let i = 1; i < n - 1; i++) {
        d[i] = (delta[i-1] + delta[i]) / 2.0;
    }
    d[n-1] = delta[n-2];
    
    // Monotone adjustments to prevent overshoot/wild waves
    for (let i = 0; i < n - 1; i++) {
        if (delta[i] === 0) {
            d[i] = 0;
            d[i+1] = 0;
        } else {
            const alpha = d[i] / delta[i];
            const beta = d[i+1] / delta[i];
            const dist = alpha * alpha + beta * beta;
            if (dist > 9.0) {
                const tau = 3.0 / Math.sqrt(dist);
                d[i] = tau * alpha * delta[i];
                d[i+1] = tau * beta * delta[i];
            }
        }
    }
    
    const c = [];
    const b = [];
    for (let i = 0; i < n - 1; i++) {
        const h = x[i+1] - x[i];
        c.push((3.0 * delta[i] - 2.0 * d[i] - d[i+1]) / h);
        b.push((d[i] + d[i+1] - 2.0 * delta[i]) / (h * h));
    }
    
    return function(t) {
        if (t <= x[0]) return y[0];
        if (t >= x[n-1]) return y[n-1];
        
        let i = 0;
        while (t > x[i+1]) {
            i++;
        }
        
        const h = t - x[i];
        return y[i] + d[i] * h + c[i] * h * h + b[i] * h * h * h;
    };
}

// Mouse events on Curve Editor
function onCurveMouseDown(e) {
    const rect = el.curveCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    
    const size = state.curve.gridSize;
    const padding = state.curve.padding;
    const innerSize = size - 2 * padding;
    
    // Map click back to [0.0, 1.0] coordinates
    const nx = (mx - padding) / innerSize;
    const ny = 1.0 - (my - padding) / innerSize;
    
    const pts = state.params.curves[state.curve.channel];
    
    // Check if clicked close to an existing point (within ~8 pixels radius)
    const threshold = 10 / innerSize;
    let clickedIdx = -1;
    
    for (let i = 0; i < pts.length; i++) {
        const dist = Math.hypot(pts[i][0] - nx, pts[i][1] - ny);
        if (dist < threshold) {
            clickedIdx = i;
            break;
        }
    }
    
    if (clickedIdx !== -1) {
        // Select existing point
        state.curve.selectedPointIdx = clickedIdx;
        state.curve.dragged = true;
    } else {
        // Clicked empty area: Insert new point (cannot insert before 0.0 or after 1.0)
        if (nx > 0.0 && nx < 1.0) {
            pts.push([nx, ny]);
            // Re-sort points by x coord
            pts.sort((a, b) => a[0] - b[0]);
            // Find its new index
            state.curve.selectedPointIdx = pts.findIndex(p => p[0] === nx && p[1] === ny);
            state.curve.dragged = true;
            drawCurve();
        }
    }
}

function onCurveMouseMove(e) {
    if (!state.curve.dragged || state.curve.selectedPointIdx === -1) return;
    
    const rect = el.curveCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    
    const size = state.curve.gridSize;
    const padding = state.curve.padding;
    const innerSize = size - 2 * padding;
    
    // Calculate normalized coords
    let nx = (mx - padding) / innerSize;
    let ny = 1.0 - (my - padding) / innerSize;
    
    // Clamp to boundaries [0, 1]
    nx = Math.max(0.0, Math.min(1.0, nx));
    ny = Math.max(0.0, Math.min(1.0, ny));
    
    const pts = state.params.curves[state.curve.channel];
    const idx = state.curve.selectedPointIdx;
    
    // If it is the start point (x=0) or end point (x=1), we cannot change their x coord
    if (idx === 0) {
        pts[idx][1] = ny;
    } else if (idx === pts.length - 1) {
        pts[idx][1] = ny;
    } else {
        // Intermediate point: x coordinate must remain between the preceding and succeeding points to prevent overlap
        const minX = pts[idx-1][0] + 0.01;
        const maxX = pts[idx+1][0] - 0.01;
        
        pts[idx][0] = Math.max(minX, Math.min(maxX, nx));
        pts[idx][1] = ny;
    }
    
    drawCurve();
    triggerPreviewUpdate();
}

function onCurveMouseUp() {
    if (state.curve.dragged) {
        state.curve.dragged = false;
        // Don't deselect automatically to show which point is selected
    }
}

function onCurveRightClick(e) {
    e.preventDefault(); // Prevent standard right-click menu
    
    const rect = el.curveCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    
    const size = state.curve.gridSize;
    const padding = state.curve.padding;
    const innerSize = size - 2 * padding;
    
    const nx = (mx - padding) / innerSize;
    const ny = 1.0 - (my - padding) / innerSize;
    
    const pts = state.params.curves[state.curve.channel];
    const threshold = 10 / innerSize;
    let clickedIdx = -1;
    
    for (let i = 0; i < pts.length; i++) {
        const dist = Math.hypot(pts[i][0] - nx, pts[i][1] - ny);
        if (dist < threshold) {
            clickedIdx = i;
            break;
        }
    }
    
    // Delete the point, but NEVER delete endpoints (0 and length-1)
    if (clickedIdx > 0 && clickedIdx < pts.length - 1) {
        pts.splice(clickedIdx, 1);
        state.curve.selectedPointIdx = -1;
        drawCurve();
        triggerPreviewUpdate();
    }
}

// Preset management UI
async function loadPresets() {
    try {
        const res = await fetch("/api/presets");
        const data = await res.json();
        state.presets = data.presets;
        
        el.presetsList.innerHTML = "";
        state.presets.forEach(p => {
            const btn = document.createElement("button");
            btn.className = "preset-item";
            btn.textContent = p;
            btn.addEventListener("click", () => applyPreset(p));
            el.presetsList.appendChild(btn);
        });
    } catch (e) {
        console.error("Failed to load presets", e);
    }
}

async function applyPreset(presetName) {
    try {
        updateStatus(`Applying preset ${presetName}...`, "pulse");
        const res = await fetch(`/api/presets/${encodeURIComponent(presetName)}`);
        if (!res.ok) throw new Error("Failed to load preset content");
        
        const preset = await res.json();
        
        // Load slider params
        state.params.exposure = preset.exposure ?? 0.0;
        state.params.contrast = preset.contrast ?? 0.0;
        state.params.highlights = preset.highlights ?? 0.0;
        state.params.shadows = preset.shadows ?? 0.0;
        state.params.saturation = preset.saturation ?? 0.0;
        state.params.temperature = preset.temperature ?? 5500.0;
        state.params.tint = preset.tint ?? 0.0;
        state.params.color_noise_reduction = preset.color_noise_reduction ?? 0.0;
        state.params.lut = preset.lut ?? "None";
        
        if (preset.curves) {
            state.params.curves = JSON.parse(JSON.stringify(preset.curves)); // deep copy
        }
        
        // Sync sliders in UI
        el.slideExposure.value = state.params.exposure;
        el.slideContrast.value = state.params.contrast;
        el.slideSaturation.value = state.params.saturation;
        el.slideHighlights.value = state.params.highlights;
        el.slideShadows.value = state.params.shadows;
        el.slideTemp.value = state.params.temperature;
        el.slideTint.value = state.params.tint;
        el.slideChromaNr.value = state.params.color_noise_reduction;
        el.lutSelect.value = state.params.lut;
        
        // Update labels
        updateSliderLabel(el.slideExposure, el.valExposure, "", 2, true);
        updateSliderLabel(el.slideContrast, el.valContrast, "", 2, true);
        updateSliderLabel(el.slideSaturation, el.valSaturation, "", 2, true);
        updateSliderLabel(el.slideHighlights, el.valHighlights, "", 2, true);
        updateSliderLabel(el.slideShadows, el.valShadows, "", 2, true);
        updateSliderLabel(el.slideTemp, el.valTemp, " K");
        updateSliderLabel(el.slideTint, el.valTint, "");
        updateSliderLabel(el.slideChromaNr, el.valChromaNr, "");
        
        // Highlight active preset in sidebar
        el.presetsList.querySelectorAll(".preset-item").forEach(btn => {
            btn.classList.toggle("active", btn.textContent === presetName);
        });
        
        // Redraw curves and refresh preview
        drawCurve();
        refreshPreview();
    } catch (e) {
        alert(`Failed to apply preset: ${e.message}`);
    }
}

async function savePreset() {
    const name = el.newPresetName.value.trim();
    if (!name) {
        alert("Please enter a preset name.");
        return;
    }
    
    // Prepare preset object
    const reqBody = {
        name: name,
        params: {
            exposure: state.params.exposure,
            contrast: state.params.contrast,
            highlights: state.params.highlights,
            shadows: state.params.shadows,
            saturation: state.params.saturation,
            temperature: state.params.temperature,
            tint: state.params.tint,
            color_noise_reduction: state.params.color_noise_reduction,
            lut: state.params.lut,
            curves: state.params.curves
        }
    };
    
    try {
        const res = await fetch("/api/presets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(reqBody)
        });
        
        if (res.ok) {
            el.newPresetName.value = "";
            loadPresets(); // refresh preset list
            updateStatus("Preset saved successfully!", "success");
        } else {
            alert("Failed to save preset");
        }
    } catch (e) {
        alert("Error saving preset: " + e.message);
    }
}

// LUTs dropdown loading
async function loadLuts() {
    try {
        const res = await fetch("/api/luts");
        const data = await res.json();
        state.luts = data.luts;
        
        el.lutSelect.innerHTML = "";
        state.luts.forEach(lut => {
            const opt = document.createElement("option");
            opt.value = lut;
            opt.textContent = lut.replace(".cube", "");
            el.lutSelect.appendChild(opt);
        });
        
        el.lutSelect.value = state.params.lut;
    } catch (e) {
        console.error("Failed to load LUT list", e);
    }
}

// Batch Exporting Flow
async function startBatchExport() {
    if (state.selectedFiles.size === 0) return;
    
    // Build parameters map: each selected file will use the CURRENT parameters
    // (In Lightroom, you apply settings to one image and batch export all selected images using those settings)
    const paramsMap = {
        default: {
            exposure: state.params.exposure,
            contrast: state.params.contrast,
            highlights: state.params.highlights,
            shadows: state.params.shadows,
            saturation: state.params.saturation,
            temperature: state.params.temperature,
            tint: state.params.tint,
            color_noise_reduction: state.params.color_noise_reduction,
            lut: state.params.lut,
            curves: state.params.curves
        }
    };
    
    // In case individual parameters were customized in a advanced version, we can write mapping:
    // For now, we apply the current visual configuration to all exported files!
    
    const filesArray = Array.from(state.selectedFiles);
    
    try {
        updateStatus("Initializing batch export...", "pulse");
        
        const res = await fetch("/api/export", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                files: filesArray,
                params_map: paramsMap
            })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Export startup failed");
        }
        
        // Show modal and start progress loop polling
        el.exportProgressFill.style.width = "0%";
        el.exportProgressPercentage.textContent = "0%";
        el.exportProgressText.textContent = `Exporting: 0 / ${filesArray.length} files`;
        el.exportCurrentFileName.textContent = "...";
        el.exportLogOutput.innerHTML = "";
        
        el.exportModal.style.display = "flex";
        el.btnCancelExport.style.display = "inline-flex";
        el.btnCloseExport.style.display = "none";
        
        // Start polling progress every 500ms
        if (state.exportStatusTimer) clearInterval(state.exportStatusTimer);
        state.exportStatusTimer = setInterval(pollExportStatus, 500);
        
    } catch (e) {
        alert("Export failed to start: " + e.message);
        updateStatus("Export start failed", "error");
    }
}

async function pollExportStatus() {
    try {
        const res = await fetch("/api/export_status");
        if (!res.ok) return;
        
        const data = await res.json();
        
        // Update percentages
        const completedCount = data.completed.length;
        const total = data.total;
        const percent = total > 0 ? Math.round((completedCount / total) * 100) : 0;
        
        el.exportProgressFill.style.width = `${percent}%`;
        el.exportProgressPercentage.textContent = `${percent}%`;
        el.exportProgressText.textContent = `Exporting: ${completedCount} / ${total} files`;
        el.exportCurrentFileName.textContent = data.current_file || "Finalizing...";
        
        // Redraw export logs
        el.exportLogOutput.innerHTML = "";
        
        // Draw logs for successes
        data.completed.forEach(name => {
            const div = document.createElement("div");
            div.className = "log-item log-success";
            div.textContent = `[SUCCESS] Exported: ${name} -> exports/${name.replace(/\.[^/.]+$/, "")}_processed.png`;
            el.exportLogOutput.appendChild(div);
        });
        
        // Draw logs for errors
        data.errors.forEach(err => {
            const div = document.createElement("div");
            div.className = "log-item log-error";
            div.textContent = `[ERROR] ${err}`;
            el.exportLogOutput.appendChild(div);
        });
        
        // Scroll log output
        el.exportLogOutput.scrollTop = el.exportLogOutput.scrollHeight;
        
        // Check if finished
        if (!data.running) {
            clearInterval(state.exportStatusTimer);
            state.exportStatusTimer = null;
            
            el.btnCancelExport.style.display = "none";
            el.btnCloseExport.style.display = "inline-flex";
            
            updateStatus(`Batch export completed (${completedCount} success, ${data.errors.length} errors)`, "success");
            
            // Add a summary line in log
            const summary = document.createElement("div");
            summary.style.fontWeight = "bold";
            summary.style.marginTop = "8px";
            summary.textContent = `[DONE] Batch completed. Files written to scanned directory exports/ subfolder.`;
            el.exportLogOutput.appendChild(summary);
            el.exportLogOutput.scrollTop = el.exportLogOutput.scrollHeight;
        }
    } catch (e) {
        console.error("Error polling export progress:", e);
    }
}

async function cancelBatchExport() {
    try {
        await fetch("/api/export_cancel", { method: "POST" });
        clearInterval(state.exportStatusTimer);
        state.exportStatusTimer = null;
        
        el.exportModal.style.display = "none";
        updateStatus("Export batch cancelled", "warning");
    } catch (e) {
        console.error("Cancel failed", e);
    }
}

// Global visual status banner
function updateStatus(text, type) {
    el.statusText.textContent = text;
    
    // Clear old status class styles
    el.statusText.parentElement.className = "global-status";
    
    const icon = el.statusText.previousElementSibling;
    icon.className = ""; // clear old lucide pulse class
    
    if (type === "pulse") {
        el.statusText.parentElement.classList.add("status-pulse");
        icon.className = "pulse-icon";
        icon.setAttribute("data-lucide", "circle-dot");
    } else if (type === "success") {
        el.statusText.parentElement.classList.add("status-success");
        icon.className = "text-success";
        icon.setAttribute("data-lucide", "check-circle");
    } else if (type === "error") {
        el.statusText.parentElement.classList.add("status-error");
        icon.className = "text-danger";
        icon.setAttribute("data-lucide", "alert-circle");
    } else if (type === "warning") {
        el.statusText.parentElement.classList.add("status-warning");
        icon.className = "text-warning";
        icon.setAttribute("data-lucide", "alert-triangle");
    }
    
    // Refresh Lucide icon rendering
    lucide.createIcons();
}
