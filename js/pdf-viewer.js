// launching PDF-Viewer
var previewFile = {
    content:{location: {url: null}},
    metaData: {fileName: null}
};

var previewConfig = {
    defaultViewMode: "FIT_WIDTH",
    showAnnotationTools: false,
    dockPageControls: false,
    showPrintPDF: false};

$("#pdfViewer").on('shown.bs.modal', function (event) {
    $("#modal-label").html(event.relatedTarget.getAttribute("data-modal-title"));

    let filePath = event.relatedTarget.href;
    previewFile.content.location.url = filePath;
    previewFile.metaData.fileName = filePath.split("/").pop();

    let adobeDCView = new AdobeDC.View({clientId: "85e4fb3e61b54f668ab5032c88be400d", divId: "adobe-dc-view"});
    adobeDCView.previewFile(previewFile, previewConfig);
});
