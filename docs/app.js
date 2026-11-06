"use strict";

const commands = {
  install: [
    'git clone https://github.com/PaoShou-China/DSSDL.git\ncd DSSDL\npython -m pip install -e ".[experiments]"\ndssdl doctor',
    "installNote",
  ],
  verify: ["dssdl verify\ndssdl check\ndssdl list", "verifyNote"],
  run: [
    "dssdl run wasserstein --name w-example\n\ndssdl run online --system fhn \\\n  --method pysr44 --seed 0 --smoke \\\n  --name online-example",
    "runNote",
  ],
};
const colors = [
  "#01519a",
  "#7f8e99",
  "#087e85",
  "#c74635",
  "#6a81a5",
  "#739383",
];
let system = "lorenz";
let language = "zh";
let currentCommand = "install";
let copyStatus = "";
let playing = false;
let lastTime = 0;
let progress = 1;
const canvas = document.querySelector("#trajectory");
const slider = document.querySelector("#time");
const play = document.querySelector("#play");
const data = window.DSSDL_RESULTS;

function translate(key) {
  return window.DSSDL_TRANSLATIONS[key][language === "en" ? 1 : 0];
}
function renderCommand() {
  const [text, noteKey] = commands[currentCommand];
  document.querySelector("#command-text").textContent = text;
  document.querySelector("#command-note").textContent = translate(noteKey);
  document.querySelector("#copy-status").textContent = copyStatus
    ? translate(copyStatus)
    : "";
}
function setLanguage(next, persist = true) {
  language = next === "en" ? "en" : "zh";
  document.documentElement.lang = language === "en" ? "en" : "zh-CN";
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = translate(element.dataset.i18n);
  });
  for (const attribute of ["aria-label", "title", "alt"]) {
    document.querySelectorAll(`[data-i18n-${attribute}]`).forEach((element) => {
      element.setAttribute(
        attribute,
        translate(element.getAttribute(`data-i18n-${attribute}`)),
      );
    });
  }
  document.querySelectorAll("[data-language]").forEach((button) => {
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.language === language),
    );
  });
  document.querySelector('meta[name="description"]').content =
    translate("description");
  document.querySelector('meta[property="og:description"]').content =
    translate("description");
  document.querySelector('meta[property="og:title"]').content =
    translate("pageTitle");
  if (persist) {
    try {
      localStorage.setItem("dssdl-language", language);
    } catch {
      // Language switching still works when browser storage is disabled.
    }
  }
  renderCommand();
  playbackState();
  renderSystem();
}
document.querySelectorAll("[data-language]").forEach((button) => {
  button.addEventListener("click", () => setLanguage(button.dataset.language));
});

function icons() {
  if (window.lucide) window.lucide.createIcons();
}
function renderPlot() {
  const record = data[system];
  const width = canvas.parentElement.clientWidth;
  const height = canvas.parentElement.clientHeight;
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  const all = record.reference.concat(
    record.methods.dssdl,
    record.methods.sindy,
  );
  let xmin = Infinity,
    xmax = -Infinity,
    ymin = Infinity,
    ymax = -Infinity;
  for (const p of all) {
    xmin = Math.min(xmin, p[0]);
    xmax = Math.max(xmax, p[0]);
    ymin = Math.min(ymin, p[1]);
    ymax = Math.max(ymax, p[1]);
  }
  const dx = (xmax - xmin) * 0.09,
    dy = (ymax - ymin) * 0.09;
  xmin -= dx;
  xmax += dx;
  ymin -= dy;
  ymax += dy;
  const m = { l: 43, r: 17, t: 18, b: 35 };
  const X = (x) => m.l + ((x - xmin) / (xmax - xmin)) * (width - m.l - m.r);
  const Y = (y) =>
    height - m.b - ((y - ymin) / (ymax - ymin)) * (height - m.t - m.b);
  ctx.font = '10px "Segoe UI", Arial';
  ctx.lineWidth = 0.8;
  for (let i = 0; i <= 4; i++) {
    const x = xmin + ((xmax - xmin) * i) / 4,
      y = ymin + ((ymax - ymin) * i) / 4;
    ctx.strokeStyle = "#dbe3e9";
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(X(x), m.t);
    ctx.lineTo(X(x), height - m.b);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(m.l, Y(y));
    ctx.lineTo(width - m.r, Y(y));
    ctx.stroke();
    ctx.fillStyle = "#71818d";
    ctx.textAlign = "center";
    ctx.fillText(x.toFixed(system === "fhn" ? 1 : 0), X(x), height - m.b + 17);
    ctx.textAlign = "right";
    ctx.fillText(y.toFixed(system === "fhn" ? 1 : 0), m.l - 8, Y(y) + 3);
  }
  ctx.setLineDash([]);
  ctx.strokeStyle = "#a6b6c1";
  ctx.strokeRect(m.l, m.t, width - m.l - m.r, height - m.t - m.b);
  const count = Math.max(2, Math.ceil(record.reference.length * progress));
  for (const [points, color, dash] of [
    [record.reference, "#87949e", [4, 3]],
    [record.methods.sindy, "#087e85", []],
    [record.methods.dssdl, "#01519a", []],
  ]) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = color === "#01519a" ? 1.7 : 1.1;
    ctx.setLineDash(dash);
    points
      .slice(0, count)
      .forEach((p, i) =>
        i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])),
      );
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.fillStyle = "#3c566b";
  ctx.font = "12px Georgia";
  ctx.textAlign = "center";
  ctx.fillText(record.axes[0], (width + m.l - m.r) / 2, height - 2);
  ctx.fillText(record.axes[1], 14, 20);
  document.querySelector("#time-label").value =
    (progress * record.duration).toFixed(2) + " s";
}
function renderSystem() {
  const record = data[system];
  document.querySelector("#plot-title").textContent =
    (system === "fhn" ? "FHN" : "Lorenz") +
    " · " +
    translate("fixedTrajectory");
  const max = Math.max(...record.wasserstein.map((r) => r.value || 0));
  document.querySelector("#metric-bars").replaceChildren(
    ...record.wasserstein.map((r, i) => {
      const row = document.createElement("div");
      row.className = "metric-row";
      const name = document.createElement("span");
      name.textContent =
        r.method === "w/o symbol" ? translate("noSymbol") : r.method;
      const track = document.createElement("div");
      track.className = "metric-track";
      const fill = document.createElement("div");
      fill.className = "metric-fill";
      fill.style.background = colors[i];
      fill.style.width =
        r.value === null ? "0" : Math.max(0.6, (r.value / max) * 100) + "%";
      track.append(fill);
      const value = document.createElement("span");
      value.className = "metric-value" + (r.value === null ? " failed" : "");
      value.textContent =
        r.value === null
          ? translate("failed")
          : r.value < 0.01
            ? r.value.toExponential(2)
            : r.value.toFixed(4);
      row.append(name, track, value);
      return row;
    }),
  );
  document.querySelector("#result-context").textContent = translate(
    system === "lorenz" ? "lorenzContext" : "fhnContext",
  );
  document.querySelector("#source-link").href =
    "https://github.com/PaoShou-China/DSSDL/tree/main/" + record.source;
  renderPlot();
}
document.querySelectorAll("[data-system]").forEach((button) =>
  button.addEventListener("click", () => {
    system = button.dataset.system;
    document
      .querySelectorAll("[data-system]")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    renderSystem();
  }),
);
document.querySelectorAll("[data-command]").forEach((button) =>
  button.addEventListener("click", () => {
    document
      .querySelectorAll("[data-command]")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    currentCommand = button.dataset.command;
    copyStatus = "";
    renderCommand();
  }),
);
document.querySelector("#copy").addEventListener("click", async () => {
  const text = document.querySelector("#command-text").textContent;
  try {
    if (navigator.clipboard && window.isSecureContext)
      await navigator.clipboard.writeText(text);
    else {
      const input = document.createElement("textarea");
      input.value = text;
      input.style.position = "fixed";
      input.style.left = "-10000px";
      document.body.append(input);
      input.select();
      const copied = document.execCommand("copy");
      input.remove();
      if (!copied) throw new Error("clipboard unavailable");
    }
    copyStatus = "copied";
  } catch {
    copyStatus = "copyFailed";
  }
  renderCommand();
});
function playbackState() {
  const label = translate(playing ? "pause" : "play");
  play.setAttribute("aria-label", label);
  play.title = label;
  play.innerHTML = playing
    ? '<i data-lucide="pause" aria-hidden="true"></i>'
    : '<i data-lucide="play" aria-hidden="true"></i>';
  icons();
}
function frame(timestamp) {
  if (!playing) return;
  if (lastTime) progress += (timestamp - lastTime) / 9000;
  lastTime = timestamp;
  if (progress >= 1) {
    progress = 1;
    playing = false;
    playbackState();
  }
  slider.value = Math.round(progress * 100);
  renderPlot();
  if (playing) requestAnimationFrame(frame);
}
play.addEventListener("click", () => {
  playing = !playing;
  if (playing && progress >= 1) progress = 0.01;
  lastTime = 0;
  playbackState();
  if (playing) requestAnimationFrame(frame);
});
slider.addEventListener("input", () => {
  playing = false;
  playbackState();
  progress = Number(slider.value) / 100;
  renderPlot();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    playing = false;
    playbackState();
  }
});
new ResizeObserver(renderPlot).observe(canvas.parentElement);
const requestedLanguage = new URLSearchParams(location.search).get("lang");
let savedLanguage;
try {
  savedLanguage = localStorage.getItem("dssdl-language");
} catch {
  savedLanguage = null;
}
setLanguage(requestedLanguage || savedLanguage || "zh", false);
