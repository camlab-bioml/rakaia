
function setupTableDownload(buttonId, tableId, filename) {
    const button = document.getElementById(buttonId);
    if (!button) return;

    if (button.dataset.listenerAttached === "true") return;

    button.addEventListener("click", function () {

        const table = document.getElementById(tableId);
        // allow for empty tables when the DOM loads i.e. panels
        //if (!table) return;

        const headers = Array.from(table.querySelectorAll("th")).map(th => th.innerText.trim());
        const rows = Array.from(table.querySelectorAll("tr"))
            .slice(1)
            .map(row =>
                Array.from(row.querySelectorAll("td")).map(cell => cell.innerText.trim())
            );

        const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();

        setTimeout(() => URL.revokeObjectURL(url), 1000);
    });

    button.dataset.listenerAttached = "true";
}


const download_observer = new MutationObserver(() => {
    setupTableDownload("download-region-statistics", "selected-area-table", "region_stats.csv")
    setupTableDownload("btn-download-panel", "imc-panel-editable", "panel.csv")
    // download_observer.disconnect();
});

download_observer.observe(document.getElementById("react-entry-point"), { childList: true, subtree: true });
