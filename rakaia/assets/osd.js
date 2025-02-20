
function toggleNavigator(display) {
    // toggle the display visibility attribute between inline block and none
    let displayReturn = display == "none" ? "inline-block": "none"
    return displayReturn;
}

// do not run as async because it is not in a module to be compatible with dash
function checkStatus(url) {
    let response = fetch(url, { method: 'HEAD' });
    let tileReturn = [400, 404, 500].includes(response.status) ? null: url
    return tileReturn;
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
        tileSources: initialTileSource
    });
    return viewer;
};

function observeCoordChange(mutationsList, viewer) {
    return new MutationObserver((mutationsList) => {
        for (let mutation of mutationsList) {
        try {
            const coordHolder = document.getElementById("transfer_coordinates").innerText;
            const [x, y, width, height] = coordHolder.split(",").map(Number);
            let imageRect = new OpenSeadragon.Rect(x, y, width, height);
            let viewportBounds = viewer.viewport.imageToViewportRectangle(imageRect);
            viewer.viewport.goHome();
            viewer.viewport.fitBounds(viewportBounds);
        } catch (error) {
        viewer.viewport.goHome();
        alert(error);
        }
        }
    });
}

const observer = new MutationObserver(() => {
    const initialTileSource = checkStatus('/static/coregister.dzi');
    viewer = renderOSDCanvas(initialTileSource);
    observer.disconnect();
    document.getElementById("update-coregister").addEventListener('click', function(e) {
    // get the unique client key from flask used to serve the static folder
    const session_id = document.getElementById("session_id").innerText;
    const newPath = `/static/coregister_${session_id}.dzi`
    //viewer = renderOSDCanvas(initialTileSource);
    const newTileSource = checkStatus(newPath);
    viewer.open(newTileSource);
    });
    if (performance.navigation.type == performance.navigation.TYPE_RELOAD) {
        viewer.open(null);}

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
