# FAQ

## Q: What are the options for downloading images from a session?

Images can be downloaded in a few ways within ccramic.

For exporting to tiff, in the left-side `Inputs/Downloads` tab under `Downloads`,
there is a collapsible button `Show download links`. Opening this collapsible
will allow you to download the canvas as a tiff file using the second link. **Important**: The current canvas
is maintained as a tiff in the backend for export, so this version will not have the
annotations saved to it (channel legend and scalebar).

If the annotations are required, the canvas can also be exported as a PNG
file with native functionality from the dash graph. Simply hover over the
current canvas and a long pop-up menu will show up in the top right, similar to what is
shown below:

<p align="center">
    <img src="assets/graph_hover.png">
</p>

Selecting the leftmost camera icon `Download plot as a png` will export the canvas
in this format. It is important to note however that the canvas is stored
natively as an encoded base64 object, so there can be distortions or
changes to the resolution of the image due to the conversion. This applies
largely to the ratio in size and location of the annotations relative to the
graph borders, which may not appear exactly as they appear in the
canvas due to canvas sizing.

If the canvas is required directly as is in PNG format, the best option is to expand the
canvas size to the appropriate dimensions and take a screenshot. This
will maintain the ratio of the annotations to the canvas borders.

## Q: When I expand the download button and click the links, nothing happens.

Certain browsers may block unauthorized downloads from unprotected
sites such as local https addresses by default, or  from certain ports.
To bypass these restrictions, right-click on the link and select `Copy link address`.
Open up a new tab and paste the file path address and click Enter. This should allow
the download to proceed as expected.
