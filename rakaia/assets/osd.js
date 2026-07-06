
function toggleNavigator(display) {
    // toggle the display visibility attribute between inline block and none
    let displayReturn = display == "none" ? "inline-block": "none"
    return displayReturn;
}

// do not run as async because it is not in a module to be compatible with dash
function checkStatus(url) {
    let response = fetch(url, { method: 'HEAD' });
    let tileReturn = [400, 404, 500, null].includes(response.status) ? null: url
    return tileReturn;
    }

function fractionToViewportZoom(fraction, minZoom, maxZoom) {
        // map the zoom as a normalized fraction of the UI zoom, not proportional to magnification level
        fraction = Math.min(Math.max(fraction, 0), 1);
        return minZoom * Math.pow(maxZoom / minZoom, fraction);
      };

const renderOSDCanvas = (initialTileSource) => {
const viewer = OpenSeadragon({
        id: "openseadragon-container",
        crossOriginPolicy: "Anonymous",
        prefixUrl: "https://openseadragon.github.io/openseadragon/images/",
        debug: false,
        showNavigator: true,
        navigatorAutoFade:  false,
        ajaxWithCredentials: false,
        showRotationControl: true,
        tileSources: initialTileSource
    });
    return viewer;
};

function renderTiles(viewer) {
    // get the unique client key from flask used to serve the static folder
    const session_id = document.getElementById("session_id").innerText;
    const newPath = `/static/coregister_${session_id}.dzi`
    //viewer = renderOSDCanvas(initialTileSource);
    const newTileSource = checkStatus(newPath);
    viewer.open(newTileSource);
    }

function observeCoordChange(mutationsList, viewer) {
    return new MutationObserver((mutationsList) => {
        for (let mutation of mutationsList) {
        try {
            const coordHolder = document.getElementById("transfer_coordinates").innerText;
            let [x, y, width, height] = coordHolder.split(",").map(Number);
            let imageRect = new OpenSeadragon.Rect(x, y, width, height);
            let viewportBounds = viewer.viewport.imageToViewportRectangle(imageRect);
            viewer.viewport.goHome();
            viewer.viewport.fitBounds(viewportBounds);
        } catch (error) {
        viewer.viewport.goHome();
        }
        }
    });
}

function observeTilesUpdated(mutationsList, viewer) {
    return new MutationObserver((mutationsList) => {
        for (let mutation of mutationsList) {
        try {
        renderTiles(viewer);
        } catch (error) {
        viewer.open(null)};
        }
    });
}

function setWSIZoomLevel(viewer) {
    const zoomValue = document.getElementById("wsi-zoom-level").value;
        if (!Number.isNaN(zoomValue) && zoomValue >= 0 && zoomValue <= 1) {
        let zoomLevel = null;
        if (zoomValue == 0) {
           viewer.viewport.goHome();
        } else {

        let tiledImage = viewer.world.getItemAt(0);

        // get type of zoom
        const zoomElement = document.getElementById("wsi-zoom-scale");
        const checkScale = zoomElement.querySelector('input[type="checkbox"]');
        const zoomScale = checkScale ? checkScale.checked : false;

        if (zoomScale) {
            // zooming that mimics microscopy magnification levels
            zoomLevel = tiledImage.imageToViewportZoom(zoomValue);
        } else {
            // this doesn't match microscopy magnification but equal viewport increments
            const minZoom = viewer.viewport.getHomeZoom();
            const maxZoom = tiledImage.imageToViewportZoom(1);
            zoomLevel = fractionToViewportZoom(zoomValue, minZoom, maxZoom);
        }
        viewer.viewport.zoomTo(zoomLevel);

        }
        }
}

const observer = new MutationObserver(() => {

    const initialTileSource = checkStatus('/static/coregister.dzi');
    const viewer = renderOSDCanvas(initialTileSource);
    observer.disconnect();

    document.getElementById("update-coregister").addEventListener('click', function(e) {
    renderTiles(viewer);
    });
    if (performance.navigation.type == performance.navigation.TYPE_RELOAD) {
        viewer.open(null);}

    viewer.addHandler('open-failed', () => {
      let el = document.querySelector('.openseadragon-message');
      el.style = 'display:none;';
    });

    viewer.addHandler('animation', function() {

    const viewport = viewer.viewport;

    // IMP: this does not take into account rotation, the coordinates stay in the original orientation space
    // Helps to map the bounding boxes back to the original image, potentially for patch extraction
    const viewportBounds = viewport.getBounds();

    const imageBounds = viewport.viewportToImageRectangle(viewportBounds);

    const minX = imageBounds.x;
    const minY = imageBounds.y;
    const maxX = imageBounds.x + imageBounds.width;
    const maxY = imageBounds.y + imageBounds.height;

    const boundsString = `X: (${minX.toFixed(0)}, ${maxX.toFixed(0)}),\nY: (${minY.toFixed(0)}, ${maxY.toFixed(0)})`;
    document.getElementById("osd-viewport-coord").innerText = boundsString;
    });

    document.getElementById("toggle-osd-navigator").addEventListener('click', function(e) {
    viewer.navigator.element.style.display = toggleNavigator(viewer.navigator.element.style.display)
    });

    document.getElementById("wsi-zoom-scale").addEventListener("change", function(e) {
            //console.log("Checkbox changed:", e.target.checked)
            setWSIZoomLevel(viewer);
    });

    document.getElementById("wsi-zoom-level").addEventListener("input", function(e) {
        setWSIZoomLevel(viewer);
        });

    const coordTransfer = document.getElementById("transfer_coordinates")
    const coordChange = observeCoordChange(coordTransfer, viewer)
    coordChange.observe(coordTransfer, {
    characterData: true, // Detect changes inside text nodes
    subtree: true, // Watch for changes inside child elements
    childList: true // Detect additions/removals of child nodes
    });

    const tilesUpdate = document.getElementById("tiles_updated")
    const tilesListener = observeTilesUpdated(tilesUpdate, viewer)
    tilesListener.observe(tilesUpdate, {
    characterData: true,
    subtree: true,
    childList: true
    });

});

observer.observe(document.getElementById("react-entry-point"), { childList: true, subtree: true });
