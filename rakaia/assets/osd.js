
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

function rotateRect(rect, angleDegrees, imageWidth, imageHeight) {
    const angle = -angleDegrees * Math.PI / 180; // note the negative sign
    const cx = imageWidth / 2;
    const cy = imageHeight / 2;

    // center of the rect
    const rx = rect.x + rect.width / 2;
    const ry = rect.y + rect.height / 2;

    // rotate center around image center
    const nx = Math.cos(angle) * (rx - cx) - Math.sin(angle) * (ry - cy) + cx;
    const ny = Math.sin(angle) * (rx - cx) + Math.cos(angle) * (ry - cy) + cy;

    return new OpenSeadragon.Rect(
        nx - rect.width / 2,
        ny - rect.height / 2,
        rect.width,
        rect.height
    );
}


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

function observeCoordChange(mutationsList, viewer) {
    return new MutationObserver((mutationsList) => {
        for (let mutation of mutationsList) {
        try {
            const coordHolder = document.getElementById("transfer_coordinates").innerText;
            let [x, y, width, height] = coordHolder.split(",").map(Number);
            const useImageCoords = document.getElementById("toggle-osi-coords")?.checked;
            console.log(useImageCoords);
            if (useImageCoords) {
            let osd_rect = new OpenSeadragon.Rect(x, y, width, height);
            let rect = viewer.viewport.imageToViewportRectangle(osd_rect);
            console.log("rect with image");
            console.log(rect);
            viewer.viewport.goHome();
            viewer.viewport.fitBounds(rect);
            } else {
            console.log(x, y, width, height);
            const item = viewer.world.getItemAt(0);
            const imgSize = item.getContentSize();
            let rect = new OpenSeadragon.Rect(x, y, width, height);
            let rotatedRect = rotateRect(rect, viewer.viewport.getRotation(), imgSize.x, imgSize.y);
            console.log(rotatedRect);
            let newBound = viewer.viewport.imageToViewportRectangle(rotatedRect);
            console.log("rect with no image");
            console.log(newBound);
            viewer.viewport.goHome();
            viewer.viewport.fitBounds(newBound);
            };
        } catch (error) {
        console.log(error);
        viewer.viewport.goHome();
        }
        }
    });
}

const observer = new MutationObserver(() => {

    const initialTileSource = checkStatus('/static/coregister.dzi');
    const viewer = renderOSDCanvas(initialTileSource);
    observer.disconnect();

    document.getElementById("update-coregister").addEventListener('click', function(e) {
    // get the unique client key from flask used to serve the static folder
    const session_id = document.getElementById("session_id").innerText;
    let newPath = `/static/coregister_${session_id}.dzi`
    //viewer = renderOSDCanvas(initialTileSource);
    const newTileSource = checkStatus(newPath);
    viewer.open(newTileSource);
    });
    if (performance.navigation.type == performance.navigation.TYPE_RELOAD) {
        viewer.open(null);}

    viewer.addHandler('open-failed', () => {
      let el = document.querySelector('.openseadragon-message');
      el.style = 'display:none;';
    });

    document.getElementById("toggle-osd-navigator").addEventListener('click', function(e) {
    viewer.navigator.element.style.display = toggleNavigator(viewer.navigator.element.style.display)
    });

    const coordTransfer = document.getElementById("transfer_coordinates")
    const coordChange = observeCoordChange(coordTransfer, viewer)
    coordChange.observe(coordTransfer, {
    characterData: true, // Detect changes inside text nodes
    subtree: true, // Watch for changes inside child elements
    childList: true // Detect additions/removals of child nodes
});
});

observer.observe(document.getElementById("react-entry-point"), { childList: true, subtree: true });
