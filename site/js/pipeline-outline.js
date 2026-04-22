(function () {
  "use strict";

  var outline = document.getElementById("pipeline-outline");
  if (!outline) return;

  var order = [
    "pipeline-overview",
    "section-segmentation",
    "section-placement",
    "section-stylization",
    "section-blending",
  ];

  var links = outline.querySelectorAll("a[data-section]");

  function setActive(id) {
    links.forEach(function (link) {
      var sid = link.getAttribute("data-section");
      if (sid === id) {
        link.classList.add("is-active");
        link.setAttribute("aria-current", "location");
      } else {
        link.classList.remove("is-active");
        link.removeAttribute("aria-current");
      }
    });
  }

  function activeFromScroll() {
    var mark = 110;
    var best = order[0];
    for (var i = 0; i < order.length; i++) {
      var el = document.getElementById(order[i]);
      if (!el) continue;
      if (el.getBoundingClientRect().top < mark) {
        best = order[i];
      }
    }
    setActive(best);
  }

  var ticking = false;
  function onScroll() {
    if (!ticking) {
      window.requestAnimationFrame(function () {
        activeFromScroll();
        ticking = false;
      });
      ticking = true;
    }
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  activeFromScroll();
})();
