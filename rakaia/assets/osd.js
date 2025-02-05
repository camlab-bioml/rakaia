const observer = new MutationObserver(() => {
    let viewer = OpenSeadragon({
    id: "openseadragon-container",
    prefixUrl: "/assets/images/",
    tileSources: "/dzi/image.dzi"});

    observer.disconnect();
});

observer.observe(document.getElementById("react-entry-point"), { childList: true, subtree: true });
