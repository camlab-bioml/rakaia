
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
        debug: true,
        showNavigator:  true,
        ajaxWithCredentials: false,
    tileSources: initialTileSource
    });
    return viewer;
};

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
});

observer.observe(document.getElementById("react-entry-point"), { childList: true, subtree: true });
