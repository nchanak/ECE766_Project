(function () {
  "use strict";

  const input = document.getElementById("photo-upload");
  const btn = document.getElementById("generate-btn");
  const btnStylize = document.getElementById("btnStylize");
  const previewWrap = document.getElementById("upload-preview-wrap");
  const previewImg = document.getElementById("upload-preview-img");
  const fileNameEl = document.getElementById("upload-file-name");
  const statusEl = document.getElementById("upload-status");

  let objectUrl = null;

  function apiBase() {
    if (typeof window.WALDO_API_BASE === "string" && window.WALDO_API_BASE.length) {
      return window.WALDO_API_BASE.replace(/\/$/, "");
    }
    return "";
  }

  function clearPreview() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    previewImg.removeAttribute("src");
    previewWrap.hidden = true;
    fileNameEl.textContent = "";
    statusEl.textContent = "";
    if (btnStylize) btnStylize.hidden = true;
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
    if (btnStylize) btnStylize.hidden = false;
    statusEl.textContent = "Image ready. Click “Stylize & play” to run the local pipeline (server must be running).";
  });

  if (btnStylize) {
    btnStylize.addEventListener("click", function () {
      const file = input.files && input.files[0];
      if (!file) {
        statusEl.textContent = "Choose an image first.";
        return;
      }
      if (typeof window.waldoAddGeneratedLevel !== "function") {
        statusEl.textContent = "Game is not ready (waldoAddGeneratedLevel missing).";
        return;
      }

      const form = new FormData();
      form.append("image", file);

      btnStylize.disabled = true;
      statusEl.textContent = "Running pipeline (this can take several minutes)…";

      const base = apiBase();
      const url = (base || "") + "/api/generate";

      fetch(url, { method: "POST", body: form })
        .then(function (res) {
          return res.json().then(function (data) {
            if (!res.ok) {
              throw new Error((data && data.error) || res.statusText);
            }
            return data;
          });
        })
        .then(function (data) {
          if (!data || !data.ok) {
            throw new Error((data && data.error) || "Unknown error");
          }
          const path = data.image;
          if (!path) throw new Error("Response missing image URL");
          const abs = new URL(path, window.location.origin).href;
          const ok = window.waldoAddGeneratedLevel({
            title: "Generated",
            src: abs,
            waldo: data.waldo,
            hitRadius: data.hitRadius,
          });
          if (ok) {
            statusEl.textContent = "Ready — find Waldo! (Heuristic target; use Show answer to check.)";
          } else {
            statusEl.textContent = "Server OK but game rejected level payload.";
          }
        })
        .catch(function (err) {
          statusEl.textContent =
            "Stylize failed: " + (err && err.message ? err.message : String(err)) +
            " — is `python waldo_game_server.py` running at this origin?";
        })
        .finally(function () {
          btnStylize.disabled = false;
        });
    });
  }
})();
