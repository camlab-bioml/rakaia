const observer = new MutationObserver(() => {
    let viewer = OpenSeadragon({
    id: "openseadragon-container",
        crossOriginPolicy: "Anonymous",
        prefixUrl: "https://openseadragon.github.io/openseadragon/images/",
        ajaxWithCredentials: false,
    tileSources: {
        type: 'image',
        url: '/static/coregister.png'
    }
    });
    observer.disconnect();
    document.getElementById("update-coregister").addEventListener('click', function(e) {
    viewer.open({
        type: 'image',
        url: '/static/coregister.png'
    });
    });
});

observer.observe(document.getElementById("react-entry-point"), { childList: true, subtree: true });
