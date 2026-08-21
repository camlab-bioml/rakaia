"""
Contains the classes and functions for rendering a GDC OSD template for a TCGA slide
"""

_OSD_GDC_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GDC Slide Viewer</title>
<script>
  if (typeof window.console === "undefined") window.console = {};
  if (typeof console.assert !== "function") {
    console.assert = function (condition) {
      if (!condition) {
        var args = Array.prototype.slice.call(arguments, 1);
        (console.error || function () {}).apply(console, ["Assertion failed:"].concat(args));
      }
    };
  }
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/openseadragon.min.js" crossorigin="anonymous"></script>
<style>
  :root {
    --bg: #14181c;
    --panel: #1c2228;
    --border: #2a323a;
    --text: #dfe6ec;
    --muted: #8a97a3;
    --accent: #4fb0ae;
    --danger: #d97757;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }

  #app { display: flex; flex-direction: column; height: 100%; }

  #statusBar {
    padding: 6px 14px; background: var(--panel); border-bottom: 1px solid var(--border);
    font-size: 12px; color: var(--muted); font-family: monospace;
  }
  #statusBar.error { color: var(--danger); }

  #viewer { flex: 1; background: #05070a; position: relative; }
  #viewer .openseadragon-canvas { background: #05070a !important; }

  #meta {
    padding: 4px 14px; background: var(--panel); border-top: 1px solid var(--border);
    font-size: 11px; color: var(--muted); font-family: monospace; display: flex; gap: 18px; flex-wrap: wrap;
  }

  .bbox-overlay {
    border: 2px solid var(--danger);
    background: rgba(217, 119, 87, 0.12);
    box-shadow: 0 0 0 1px rgba(0,0,0,0.6);
    pointer-events: none;
  }
</style>
</head>
<body>
<div id="app">
  <div id="statusBar">Loading slide __FILE_ID__…</div>
  <div id="viewer"></div>
  <div id="meta"></div>
</div>

<script>
const TILE_BASE = "https://portal.gdc.cancer.gov/auth/api/v0/tile";

// Values supplied by the Dash callback (baked in server-side).
const FILE_ID = "__FILE_ID__";
const BOX_X = __X__;
const BOX_Y = __Y__;
const BOX_N = __N__;

class GDCTileSource extends OpenSeadragon.TileSource {
  constructor(fileId, metadata) {
    const width = parseInt(metadata.Width, 10);
    const height = parseInt(metadata.Height, 10);
    const tileSize = parseInt(metadata.TileSize, 10);
    const overlap = parseInt(metadata.Overlap, 10) || 0;
    const maxLevel = Math.ceil(Math.log2(Math.max(width, height)));

    super({ width, height, tileSize, tileOverlap: overlap, minLevel: 0, maxLevel });
    this.fileId = fileId;
  }

  getTileUrl(level, x, y) {
    return `${TILE_BASE}/${this.fileId}?level=${level}&x=${x}&y=${y}`;
  }
}

window.addEventListener("error", (e) => {
  console.error("window error:", e.error || e.message, e);
});
window.addEventListener("unhandledrejection", (e) => {
  console.error("unhandled rejection:", e.reason);
  setStatus(`Unhandled error: ${e.reason && e.reason.message ? e.reason.message : e.reason}`, true);
});

const viewer = OpenSeadragon({
  id: "viewer",
  prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
  showNavigator: true,
  visibilityRatio: 1,
  minZoomImageRatio: 0.8,
  crossOriginPolicy: false,
});

viewer.addHandler("open-failed", (e) => {
  console.error("OpenSeadragon open-failed:", e);
  setStatus(`OpenSeadragon failed to open the tile source: ${e.message || "unknown error"}`, true);
});
viewer.addHandler("tile-load-failed", (e) => {
  console.error("Tile load failed:", e.tile && e.tile.url, e);
});

const statusEl = document.getElementById("statusBar");
const metaEl = document.getElementById("meta");

function setStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.className = isError ? "error" : "";
}

function drawBox(x, y, n) {
  const tiledImage = viewer.world.getItemAt(0);
  if (!tiledImage) return;
  const rect = tiledImage.imageToViewportRectangle(x - n / 2, y - n / 2, n, n);
  const el = document.createElement("div");
  el.className = "bbox-overlay";
  viewer.addOverlay({ element: el, location: rect });
  viewer.viewport.fitBounds(rect, true);
}

async function loadSlide(fileId, x, y, n) {
  setStatus(`Fetching metadata for ${fileId}…`, false);
  try {
    let metaResp;
    try {
      metaResp = await fetch(`${TILE_BASE}/metadata/${fileId}`);
    } catch (networkErr) {
      throw new Error(
        `Network/CORS error reaching ${TILE_BASE}/metadata/${fileId} — ` +
        `check the Network tab; if it shows "(blocked:cors)", the host ` +
        `page's origin isn't allowed by portal.gdc.cancer.gov.`
      );
    }
    if (!metaResp.ok) throw new Error(`Metadata request failed: HTTP ${metaResp.status}`);
    const metadata = await metaResp.json();
    if (!metadata.Width || !metadata.Height || !metadata.TileSize) {
      throw new Error("Metadata response missing Width/Height/TileSize.");
    }

    const tileSource = new GDCTileSource(fileId, metadata);
    viewer.open(tileSource);
    viewer.addOnceHandler("open", () => drawBox(x, y, n));

    metaEl.textContent =
      `width=${metadata.Width}  height=${metadata.Height}  ` +
      `tileSize=${metadata.TileSize}  overlap=${metadata.Overlap}  ` +
      `maxLevel=${tileSource.maxLevel}  box=(${x}, ${y}) n=${n}`;
    setStatus(`Loaded ${fileId}`, false);
  } catch (err) {
    console.error(err);
    setStatus(`${err.message}`, true);
  }
}

loadSlide(FILE_ID, BOX_X, BOX_Y, BOX_N);
</script>
</body>
</html>
"""


def build_viewer_html(file_id: str, x: float, y: float, n: float = 2500) -> str:
    """
    Build the standalone HTML/JS for the GDC OpenSeadragon viewer, with a
    fixed n x n bounding box drawn and centered on (x, y).
    """
    return (
        _OSD_GDC_TEMPLATE
        .replace("__FILE_ID__", str(file_id))
        .replace("__X__", repr(float(x)))
        .replace("__Y__", repr(float(y)))
        .replace("__N__", repr(float(n))))
