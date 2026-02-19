function bindUploadPreview(inputId, previewImgId, dropzoneId, filenameId, dropPreviewId, dropIconId) {
  const input = document.getElementById(inputId);
  const previewImg = document.getElementById(previewImgId);
  const dropzone = document.getElementById(dropzoneId);
  const filenameEl = document.getElementById(filenameId);
  const dropPreview = document.getElementById(dropPreviewId);
  const dropIcon = document.getElementById(dropIconId);
  if (!input) return;

  const wrapper = previewImg ? previewImg.closest(".upload-preview-wrap") : null;
  const updatePreview = (file) => {
    if (!file) {
      if (previewImg) {
        previewImg.removeAttribute("src");
      }
      if (wrapper) {
        wrapper.classList.remove("show");
      }
      if (filenameEl) filenameEl.textContent = "No file selected";
      if (dropPreview) dropPreview.classList.remove("show");
      if (dropIcon) dropIcon.style.display = "block";
      if (dropzone) dropzone.classList.remove("has-image");
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    if (previewImg) {
      previewImg.src = objectUrl;
    }
    if (wrapper) {
      wrapper.classList.add("show");
    }
    if (filenameEl) filenameEl.textContent = file.name;
    if (dropPreview) {
      dropPreview.src = objectUrl;
      dropPreview.classList.add("show");
    }
    if (dropIcon) dropIcon.style.display = "none";
    if (dropzone) dropzone.classList.add("has-image");
  };

  input.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    updatePreview(file);
  });

  if (dropzone) {
    dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });
    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
      const files = event.dataTransfer && event.dataTransfer.files;
      if (!files || files.length === 0) return;
      input.files = files;
      updatePreview(files[0]);
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  bindUploadPreview("analyze-image", "analyze-preview", "analyze-dropzone", "analyze-filename", "analyze-drop-preview", "analyze-drop-icon");
  bindUploadPreview("visualizer-image", "visualizer-preview", "visualizer-dropzone", "visualizer-filename", "visualizer-drop-preview", "visualizer-drop-icon");
});
