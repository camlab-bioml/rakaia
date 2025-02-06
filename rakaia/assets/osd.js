const observer = new MutationObserver(() => {
    let viewer = OpenSeadragon({
    id: "openseadragon-container",
        crossOriginPolicy: "Anonymous",
        prefixUrl: "https://openseadragon.github.io/openseadragon/images/",
        debug: true,
        showNavigator:  true,
        ajaxWithCredentials: false,
    tileSources: '/static/coregister.dzi'
    });
    observer.disconnect();
    document.getElementById("update-coregister").addEventListener('click', function(e) {
    viewer.open('/static/coregister.dzi');
    });
});

observer.observe(document.getElementById("react-entry-point"), { childList: true, subtree: true });
