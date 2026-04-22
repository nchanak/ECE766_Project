(function () {
  "use strict";

  const input = document.getElementById("photo-upload");
  const btn = document.getElementById("generate-btn");
  const previewWrap = document.getElementById("upload-preview-wrap");
  const previewImg = document.getElementById("upload-preview-img");
  const fileNameEl = document.getElementById("upload-file-name");
  const statusEl = document.getElementById("upload-status");

  let objectUrl = null;

  function clearPreview() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    previewImg.removeAttribute("src");
    previewWrap.hidden = true;
    fileNameEl.textContent = "";
    statusEl.textContent = "";
  }

  if (!input || !btn) return;

  btn.addEventListener("click", function () {
    input.click();
  });

  input.addEventListener("change", function () {
    const file = input.files && input.files[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      clearPreview();
      statusEl.textContent = "Please choose an image file.";
      return;
    }

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    previewImg.src = objectUrl;
    previewImg.alt = "Preview: " + file.name;
    previewWrap.hidden = false;
    fileNameEl.textContent = file.name;
    statusEl.textContent =
      "Image ready. Pipeline hookup will replace the scene here when available.";
  });
})();
