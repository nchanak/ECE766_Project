(function () {
  "use strict";

  const stage = document.getElementById("stage");
  const hitLayer = document.getElementById("hitLayer");
  const levelImage = document.getElementById("levelImage");
  const levelCanvas = document.getElementById("levelCanvas");
  const timerEl = document.getElementById("timer");
  const resultLine = document.getElementById("resultLine");
  const resultText = document.getElementById("resultText");
  const showAnswerEl = document.getElementById("showAnswer");
  const btnNewRound = document.getElementById("btnNewRound");
  const btnStartGame = document.getElementById("btnStartGame");
  const levelPick = document.getElementById("levelPick");
  const levelPickWrap = document.getElementById("levelPickWrap");
  const imageUpload = document.getElementById("imageUpload");
  const btnGenerateRound = document.getElementById("btnGenerateRound");
  const generatorStatus = document.getElementById("generatorStatus");

  const levels = window.WALDO_LEVELS || [];
  let levelIndex = 0;
  /** @type {typeof levels[0] & { waldo?: { x: number; y: number } }} */
  let current = null;
  let won = false;
  let timerStart = 0;
  let rafId = 0;
  let wrongTimeout = 0;
  let generating = false;

  const audioCtx = typeof AudioContext !== "undefined" ? new AudioContext() : null;

  function resumeAudio() {
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume();
    }
  }

  function beep(freq, duration, type = "sine", gain = 0.12) {
    if (!audioCtx) return;
    const t0 = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t0);
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(gain, t0 + 0.02);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + duration);
    osc.connect(g);
    g.connect(audioCtx.destination);
    osc.start(t0);
    osc.stop(t0 + duration + 0.05);
  }

  function playSuccessSound() {
    if (!audioCtx) return;
    resumeAudio();
    beep(523.25, 0.12, "sine", 0.1);
    setTimeout(() => beep(659.25, 0.12, "sine", 0.1), 100);
    setTimeout(() => beep(783.99, 0.2, "sine", 0.12), 200);
  }

  function playFailSound() {
    if (!audioCtx) return;
    resumeAudio();
    beep(120, 0.25, "sawtooth", 0.08);
    setTimeout(() => beep(90, 0.2, "sawtooth", 0.06), 120);
  }

  function setGeneratorStatus(message, tone = "") {
    if (!generatorStatus) return;
    generatorStatus.textContent = message;
    generatorStatus.classList.remove("error", "success");
    if (tone) {
      generatorStatus.classList.add(tone);
    }
  }

  function setGenerating(nextGenerating) {
    generating = nextGenerating;
    if (btnGenerateRound) btnGenerateRound.disabled = nextGenerating;
    if (imageUpload) imageUpload.disabled = nextGenerating;
  }

  function formatSeconds(ms) {
    return (ms / 1000).toFixed(1) + "s";
  }

  function getMediaSize() {
    if (current.type === "canvas") {
      return {
        w: levelCanvas.width,
        h: levelCanvas.height,
      };
    }
    return {
      w: levelImage.naturalWidth,
      h: levelImage.naturalHeight,
    };
  }

  function getContainedRect(mediaW, mediaH) {
    const cr = stage.getBoundingClientRect();
    const cw = cr.width;
    const ch = cr.height;
    if (!mediaW || !mediaH) return null;
    const scale = Math.min(cw / mediaW, ch / mediaH);
    const dw = mediaW * scale;
    const dh = mediaH * scale;
    const left = cr.left + (cw - dw) / 2;
    const top = cr.top + (ch - dh) / 2;
    return { left, top, width: dw, height: dh, scale, cr };
  }

  function clientToNormalized(clientX, clientY) {
    const { w, h } = getMediaSize();
    const rect = getContainedRect(w, h);
    if (!rect) return null;
    const ix = (clientX - rect.left) / rect.width;
    const iy = (clientY - rect.top) / rect.height;
    if (ix < 0 || ix > 1 || iy < 0 || iy > 1) return null;
    return { nx: ix, ny: iy };
  }

  function distanceToWaldo(nx, ny) {
    const { w, h } = getMediaSize();
    const wx = current.waldo.x;
    const wy = current.waldo.y;
    const dx = (nx - wx) * w;
    const dy = (ny - wy) * h;
    return Math.hypot(dx, dy);
  }

  function isHit(nx, ny) {
    const { w } = getMediaSize();
    const maxDist = current.hitRadius * w;
    return distanceToWaldo(nx, ny) <= maxDist;
  }

  function clearLayer() {
    hitLayer.replaceChildren();
  }

  function placeRing(nx, ny, className = "answer-ring") {
    const { w, h } = getMediaSize();
    const rect = getContainedRect(w, h);
    if (!rect) return;
    const el = document.createElement("div");
    el.className = className;
    const px = rect.left - rect.cr.left + nx * rect.width;
    const py = rect.top - rect.cr.top + ny * rect.height;
    el.style.left = px + "px";
    el.style.top = py + "px";
    const r = Math.max(18, current.hitRadius * rect.width * 1.35);
    el.style.width = r * 2 + "px";
    el.style.height = r * 2 + "px";
    hitLayer.appendChild(el);
  }

  function placeFeedback(nx, ny, good) {
    const { w, h } = getMediaSize();
    const rect = getContainedRect(w, h);
    if (!rect) return;
    const el = document.createElement("div");
    el.className = "feedback " + (good ? "good" : "bad");
    el.textContent = good ? "\u2714" : "\u2717";
    const px = rect.left - rect.cr.left + nx * rect.width;
    const py = rect.top - rect.cr.top + ny * rect.height;
    el.style.left = px + "px";
    el.style.top = py + "px";
    hitLayer.appendChild(el);
    return el;
  }

  function stopTimerLoop() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
  }

  function tickTimer() {
    if (won) return;
    const elapsed = performance.now() - timerStart;
    timerEl.textContent = formatSeconds(elapsed);
    rafId = requestAnimationFrame(tickTimer);
  }

  function resetRoundTimer() {
    stopTimerLoop();
    timerStart = performance.now();
    timerEl.textContent = "0.0s";
    rafId = requestAnimationFrame(tickTimer);
  }

  function stopTimerDisplayFinal() {
    stopTimerLoop();
    const elapsed = performance.now() - timerStart;
    timerEl.textContent = formatSeconds(elapsed);
    return elapsed;
  }

  function drawCrowdCanvas(waldoX, waldoY) {
    const ctx = levelCanvas.getContext("2d");
    const W = levelCanvas.width;
    const H = levelCanvas.height;
    ctx.fillStyle = "#2a3444";
    ctx.fillRect(0, 0, W, H);

    const rnd = (a, b) => a + Math.random() * (b - a);
    for (let i = 0; i < 420; i++) {
      ctx.fillStyle = `hsl(${rnd(0, 360)}, ${rnd(35, 70)}%, ${rnd(35, 65)}%)`;
      ctx.beginPath();
      ctx.arc(rnd(0, W), rnd(0, H), rnd(3, 9), 0, Math.PI * 2);
      ctx.fill();
    }

    const wx = waldoX * W;
    const wy = waldoY * H;

    ctx.save();
    ctx.translate(wx, wy);
    ctx.scale(1, 1);
    const bodyH = 38;
    const bodyW = 14;
    for (let row = 0; row < 5; row++) {
      const y = -bodyH / 2 + row * (bodyH / 5);
      ctx.fillStyle = row % 2 === 0 ? "#c41e3a" : "#f5f5f5";
      ctx.fillRect(-bodyW / 2, y, bodyW, bodyH / 5 + 0.5);
    }
    ctx.fillStyle = "#f4d0b0";
    ctx.beginPath();
    ctx.arc(0, -bodyH / 2 - 10, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#3d2c1e";
    ctx.fillRect(-7, -bodyH / 2 - 12, 14, 4);
    ctx.strokeStyle = "#222";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-bodyW / 2 - 2, -bodyH / 2 + 6);
    ctx.lineTo(-bodyW / 2 - 14, bodyH / 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(bodyW / 2 + 2, -bodyH / 2 + 6);
    ctx.lineTo(bodyW / 2 + 14, bodyH / 2);
    ctx.stroke();
    ctx.restore();
  }

  function hideLevelImage() {
    levelImage.hidden = true;
    levelImage.removeAttribute("src");
  }

  function setupCanvasLevel() {
    hideLevelImage();
    levelCanvas.hidden = false;
    const margin = 0.12;
    const waldoX = margin + Math.random() * (1 - 2 * margin);
    const waldoY = margin + Math.random() * (1 - 2 * margin);
    current.waldo = { x: waldoX, y: waldoY };
    drawCrowdCanvas(waldoX, waldoY);
  }

  function setupImageLevel() {
    return new Promise((resolve, reject) => {
      levelCanvas.hidden = true;
      levelImage.hidden = false;
      const done = () => {
        levelImage.removeEventListener("load", onLoad);
        levelImage.removeEventListener("error", onErr);
      };
      const onLoad = () => {
        done();
        resolve();
      };
      const onErr = () => {
        done();
        reject(new Error("image load failed"));
      };
      levelImage.addEventListener("load", onLoad);
      levelImage.addEventListener("error", onErr);
      levelImage.src = current.src;
      levelImage.alt = current.title || "Game scene";
    });
  }

  function updateAnswerOverlay() {
    clearLayer();
    if (!current || !current.waldo) return;
    if (won) {
      placeRing(current.waldo.x, current.waldo.y);
      placeFeedback(current.waldo.x, current.waldo.y, true);
      return;
    }
    if (showAnswerEl.checked) {
      placeRing(current.waldo.x, current.waldo.y);
    }
  }

  function showConfigError(message) {
    hitLayer.classList.remove("interactive");
    if (btnStartGame) btnStartGame.hidden = true;
    resultLine.hidden = false;
    resultText.textContent = message;
    resultText.classList.remove("success");
    resultText.classList.add("error");
  }

  function prepareLobby() {
    stopTimerLoop();
    timerEl.textContent = "0.0s";
    hitLayer.classList.remove("interactive");
    if (btnStartGame) btnStartGame.hidden = false;
  }

  function beginPlaySession() {
    resumeAudio();
    if (btnStartGame) btnStartGame.hidden = true;
    hitLayer.classList.add("interactive");
    resetRoundTimer();
  }

  async function loadLevelContent() {
    if (current.type === "canvas") {
      setupCanvasLevel();
    } else if (current.type === "image") {
      await setupImageLevel();
    } else {
      throw new Error("unknown type");
    }
  }

  async function reloadCurrentLevelForNewRound() {
    try {
      await loadLevelContent();
      clearLayer();
      updateAnswerOverlay();
      hitLayer.classList.add("interactive");
      resetRoundTimer();
      if (btnStartGame) btnStartGame.hidden = true;
    } catch {
      hideLevelImage();
      levelCanvas.hidden = true;
      showConfigError("Image failed to load. Check path: " + (current.src || ""));
    }
  }

  function applyWonUI(elapsedMs) {
    if (wrongTimeout) {
      clearTimeout(wrongTimeout);
      wrongTimeout = 0;
    }
    won = true;
    stopTimerLoop();
    timerEl.textContent = formatSeconds(elapsedMs);
    resultLine.hidden = false;
    resultText.textContent = "Found! Time: " + formatSeconds(elapsedMs);
    resultText.classList.remove("error");
    resultText.classList.add("success");
    hitLayer.classList.remove("interactive");
    playSuccessSound();
    updateAnswerOverlay();
  }

  function showWrongAttempt(nx, ny) {
    playFailSound();
    placeFeedback(nx, ny, false);
    if (wrongTimeout) clearTimeout(wrongTimeout);
    wrongTimeout = window.setTimeout(() => {
      wrongTimeout = 0;
      if (!won) {
        clearLayer();
        updateAnswerOverlay();
      }
    }, 700);
  }

  function onStageClick(ev) {
    resumeAudio();
    if (won) return;
    const pt = clientToNormalized(ev.clientX, ev.clientY);
    if (!pt) return;
    const { nx, ny } = pt;
    if (isHit(nx, ny)) {
      if (wrongTimeout) {
        clearTimeout(wrongTimeout);
        wrongTimeout = 0;
      }
      const elapsed = stopTimerDisplayFinal();
      applyWonUI(elapsed);
    } else {
      showWrongAttempt(nx, ny);
    }
  }

  function syncStageSize() {
    requestAnimationFrame(() => {
      updateAnswerOverlay();
    });
  }

  function fillLevelSelect() {
    if (!levelPick || !levelPickWrap) return;
    levelPick.replaceChildren();
    levels.forEach((L, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = L.title || L.id || "Level " + (i + 1);
      levelPick.appendChild(o);
    });
    levelPick.value = String(levelIndex);
    levelPickWrap.hidden = levels.length <= 1;
  }

  function addGeneratedLevel(level) {
    levels.push(level);
    levelIndex = levels.length - 1;
    fillLevelSelect();
    if (levelPick) levelPick.value = String(levelIndex);
  }

  async function generateRoundFromUpload() {
    if (generating) return;
    if (!imageUpload || !imageUpload.files || !imageUpload.files[0]) {
      setGeneratorStatus("Choose an image first.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageUpload.files[0]);

    setGenerating(true);
    setGeneratorStatus("Generating round... this can take a while.", "");

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Generation failed.");
      }
      addGeneratedLevel(data.level);
      setGeneratorStatus("Round generated. Starting the new level.", "success");
      await startLevel();
    } catch (err) {
      setGeneratorStatus("Generation failed: " + (err && err.message ? err.message : String(err)), "error");
    } finally {
      setGenerating(false);
    }
  }

  async function startLevel() {
    if (!levels.length) {
      hideLevelImage();
      levelCanvas.hidden = true;
      showConfigError("No levels configured. Edit levels.js.");
      return;
    }
    current = { ...levels[levelIndex] };
    won = false;
    resultLine.hidden = true;
    resultText.classList.remove("error", "success");

    if (current.type !== "canvas" && current.type !== "image") {
      hideLevelImage();
      levelCanvas.hidden = true;
      showConfigError("Unknown level type.");
      return;
    }

    try {
      await loadLevelContent();
    } catch {
      hideLevelImage();
      levelCanvas.hidden = true;
      showConfigError("Image failed to load. Check path: " + (current.src || ""));
      return;
    }

    clearLayer();
    updateAnswerOverlay();
    prepareLobby();
  }

  function newRound() {
    if (wrongTimeout) {
      clearTimeout(wrongTimeout);
      wrongTimeout = 0;
    }
    won = false;
    resultLine.hidden = true;
    resultText.classList.remove("error", "success");
    if (current && current.type === "canvas") {
      setupCanvasLevel();
      clearLayer();
      updateAnswerOverlay();
      hitLayer.classList.add("interactive");
      resetRoundTimer();
      if (btnStartGame) btnStartGame.hidden = true;
    } else if (current) {
      reloadCurrentLevelForNewRound();
    }
  }

  hitLayer.addEventListener("click", onStageClick);
  window.addEventListener("resize", syncStageSize);

  showAnswerEl.addEventListener("change", () => {
    updateAnswerOverlay();
  });

  if (btnStartGame) {
    btnStartGame.addEventListener("click", () => {
      beginPlaySession();
    });
  }

  btnNewRound.addEventListener("click", () => {
    resumeAudio();
    newRound();
  });

  document.body.addEventListener(
    "click",
    () => {
      resumeAudio();
    },
    { once: true }
  );

  if (levelPick) {
    levelPick.addEventListener("change", () => {
      resumeAudio();
      levelIndex = parseInt(levelPick.value, 10);
      if (wrongTimeout) {
        clearTimeout(wrongTimeout);
        wrongTimeout = 0;
      }
      won = false;
      resultLine.hidden = true;
      resultText.classList.remove("error", "success");
      startLevel();
    });
  }

  if (btnGenerateRound) {
    btnGenerateRound.addEventListener("click", () => {
      resumeAudio();
      generateRoundFromUpload();
    });
  }

  fillLevelSelect();
  startLevel();
})();
