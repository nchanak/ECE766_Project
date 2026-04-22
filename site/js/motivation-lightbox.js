(function () {
  "use strict";

  const lightbox = document.getElementById("motivation-lightbox");
  if (!lightbox) return;

  const img = lightbox.querySelector(".lightbox__img");
  const backdrop = lightbox.querySelector(".lightbox__backdrop");
  if (!img) return;

  function open(src, alt) {
    img.src = src;
    img.alt = alt || "";
    lightbox.hidden = false;
    document.body.style.overflow = "hidden";
  }

  function close() {
    lightbox.hidden = true;
    img.removeAttribute("src");
    img.alt = "";
    document.body.style.overflow = "";
  }

  if (backdrop) {
    backdrop.addEventListener("click", function () {
      close();
    });
  }

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !lightbox.hidden) {
      close();
    }
  });

  document.querySelectorAll(".book-thumb__enlarge").forEach(function (btn) {
    btn.addEventListener("click", function () {
      const fig = btn.closest("figure");
      const thumb = fig && fig.querySelector(".book-thumb__img");
      if (!thumb) return;
      const full = btn.getAttribute("data-full") || thumb.getAttribute("src");
      const alt = thumb.getAttribute("alt") || "";
      open(full, alt);
    });
  });
})();
