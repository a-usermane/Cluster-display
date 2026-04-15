import { runKMeans, explainPointKMeans } from "./algorithms/kmeans.js";
import { runDBSCAN, explainPointDBSCAN } from "./algorithms/dbscan.js";
import { runAgglomerative, explainPointAgglomerative } from "./algorithms/agglomerative.js";
import { renderAlgorithmSummary, renderPointDetail } from "./formula.js";

const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");
const summaryEl = document.getElementById("formulaSummary");
const pointEl = document.getElementById("pointDetail");
const legendEl = document.getElementById("legend");
const statusHintEl = document.getElementById("statusHint");

const controls = {
  numClusters: document.getElementById("numClusters"),
  pointsPerCluster: document.getElementById("pointsPerCluster"),
  stdDev: document.getElementById("stdDev"),
  spread: document.getElementById("spread"),
  seed: document.getElementById("seed"),
  algorithm: document.getElementById("algorithm"),
  kmeansK: document.getElementById("kmeansK"),
  kmeansMaxIter: document.getElementById("kmeansMaxIter"),
  kmeansTol: document.getElementById("kmeansTol"),
  dbscanEps: document.getElementById("dbscanEps"),
  dbscanMinPts: document.getElementById("dbscanMinPts"),
  aggloK: document.getElementById("aggloK"),
  kmeansParams: document.getElementById("kmeansParams"),
  dbscanParams: document.getElementById("dbscanParams"),
  aggloParams: document.getElementById("aggloParams"),
  generateBtn: document.getElementById("generateBtn"),
  clusterBtn: document.getElementById("clusterBtn"),
  randomParamsBtn: document.getElementById("randomParamsBtn"),
  resetBtn: document.getElementById("resetBtn"),
  playBtn: document.getElementById("playBtn"),
  pauseBtn: document.getElementById("pauseBtn"),
  stepBtn: document.getElementById("stepBtn"),
  speedSelect: document.getElementById("speedSelect"),
};

const DEFAULT_PARAMS = {
  numClusters: 3,
  pointsPerCluster: 80,
  stdDev: 6,
  spread: 40,
  seed: 42,
  kmeansK: 3,
  kmeansMaxIter: 50,
  kmeansTol: 0.001,
  dbscanEps: 8,
  dbscanMinPts: 6,
  aggloK: 3,
};

const palette = [
  "#e74c3c",
  "#2980b9",
  "#16a085",
  "#f39c12",
  "#8e44ad",
  "#d35400",
  "#2ecc71",
  "#1abc9c",
  "#c0392b",
  "#34495e",
];

const state = {
  dataset: [],
  currentAlgorithm: controls.algorithm.value,
  clusterResult: null,
  selectedPointId: null,
  pointExplanation: null,
  params: readParams(),
  bounds: { minX: -60, maxX: 60, minY: -60, maxY: 60 },
  animation: {
    isAnimating: false,
    frames: [],
    frameIndex: 0,
    timerId: null,
    activeCentroids: null,
  },
};

function readParams() {
  return {
    numClusters: Number(controls.numClusters.value),
    pointsPerCluster: Number(controls.pointsPerCluster.value),
    stdDev: Number(controls.stdDev.value),
    spread: Number(controls.spread.value),
    seed: Number(controls.seed.value),
    kmeansK: Number(controls.kmeansK.value),
    kmeansMaxIter: Number(controls.kmeansMaxIter.value),
    kmeansTol: Number(controls.kmeansTol.value),
    dbscanEps: Number(controls.dbscanEps.value),
    dbscanMinPts: Number(controls.dbscanMinPts.value),
    aggloK: Number(controls.aggloK.value),
  };
}

function setParams(p) {
  controls.numClusters.value = p.numClusters;
  controls.pointsPerCluster.value = p.pointsPerCluster;
  controls.stdDev.value = p.stdDev;
  controls.spread.value = p.spread;
  controls.seed.value = p.seed;
  controls.kmeansK.value = p.kmeansK;
  controls.kmeansMaxIter.value = p.kmeansMaxIter;
  controls.kmeansTol.value = p.kmeansTol;
  controls.dbscanEps.value = p.dbscanEps;
  controls.dbscanMinPts.value = p.dbscanMinPts;
  controls.aggloK.value = p.aggloK;
}

function setStatus(text) {
  statusHintEl.textContent = text;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng, mean = 0, std = 1) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = Math.max(rng(), 1e-12);
  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z0 * std + mean;
}

function generateDataset() {
  const p = readParams();
  controls.aggloK.value = String(p.numClusters);
  state.params = p;
  const rng = mulberry32(p.seed);
  const points = [];
  let id = 1;

  const centers = Array.from({ length: p.numClusters }, () => ({
    x: (rng() * 2 - 1) * p.spread,
    y: (rng() * 2 - 1) * p.spread,
  }));

  for (let c = 0; c < centers.length; c += 1) {
    for (let i = 0; i < p.pointsPerCluster; i += 1) {
      points.push({
        id: id++,
        x: gaussian(rng, centers[c].x, p.stdDev),
        y: gaussian(rng, centers[c].y, p.stdDev),
        trueCluster: c,
        predCluster: null,
      });
    }
  }

  state.dataset = points;
  state.clusterResult = null;
  state.selectedPointId = null;
  state.pointExplanation = null;
  controls.clusterBtn.disabled = points.length === 0;
  clearAnimation();
  updateAnimationButtons();
  updateBounds();
  setStatus(`已生成 ${points.length} 个点，请选择算法执行聚类。`);
  renderAll();
}

function updateBounds() {
  if (!state.dataset.length) {
    state.bounds = { minX: -60, maxX: 60, minY: -60, maxY: 60 };
    return;
  }
  const xs = state.dataset.map((p) => p.x);
  const ys = state.dataset.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padX = Math.max(10, (maxX - minX) * 0.15);
  const padY = Math.max(10, (maxY - minY) * 0.15);
  state.bounds = { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
}

function worldToScreen(x, y) {
  const { minX, maxX, minY, maxY } = state.bounds;
  const px = ((x - minX) / (maxX - minX || 1)) * canvas.width;
  const py = canvas.height - ((y - minY) / (maxY - minY || 1)) * canvas.height;
  return { x: px, y: py };
}

function drawAxesAndGrid() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#edf1f8";
  ctx.lineWidth = 1;
  for (let i = 1; i < 10; i += 1) {
    const x = (canvas.width / 10) * i;
    const y = (canvas.height / 10) * i;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  const origin = worldToScreen(0, 0);
  ctx.strokeStyle = "#9aa5bc";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(0, origin.y);
  ctx.lineTo(canvas.width, origin.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(origin.x, 0);
  ctx.lineTo(origin.x, canvas.height);
  ctx.stroke();
}

function getPointColor(point) {
  if (point.predCluster == null) {
    return "#3577e5";
  }
  if (point.predCluster === -1) {
    return "#9aa0ab";
  }
  return palette[point.predCluster % palette.length];
}

function drawPoints() {
  state.dataset.forEach((point) => {
    const p = worldToScreen(point.x, point.y);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = getPointColor(point);
    ctx.fill();

    if (state.selectedPointId === point.id) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#111";
      ctx.stroke();
    }
  });
}

function computeCentersFromLabels(labels) {
  const map = new Map();
  labels.forEach((label, idx) => {
    if (label == null || label < 0) {
      return;
    }
    const bucket = map.get(label) ?? { x: 0, y: 0, count: 0 };
    bucket.x += state.dataset[idx].x;
    bucket.y += state.dataset[idx].y;
    bucket.count += 1;
    map.set(label, bucket);
  });
  return [...map.entries()].map(([label, bucket]) => ({
    label,
    x: bucket.x / bucket.count,
    y: bucket.y / bucket.count,
  }));
}

function drawClusterCenters() {
  if (!state.dataset.length) {
    return;
  }
  let centers = [];
  if (state.currentAlgorithm === "kmeans" && state.animation.activeCentroids) {
    centers = state.animation.activeCentroids.map((c, idx) => ({ label: idx, x: c.x, y: c.y }));
  } else if (state.currentAlgorithm === "kmeans" && state.clusterResult?.centroids) {
    centers = state.clusterResult.centroids.map((c, idx) => ({ label: idx, x: c.x, y: c.y }));
  } else {
    centers = computeCentersFromLabels(state.dataset.map((p) => p.predCluster));
  }
  centers.forEach((center) => {
    const p = worldToScreen(center.x, center.y);
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(p.x - 6, p.y);
    ctx.lineTo(p.x + 6, p.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p.x, p.y - 6);
    ctx.lineTo(p.x, p.y + 6);
    ctx.stroke();
  });
}

function renderLegend() {
  const labels = state.dataset.map((p) => p.predCluster).filter((v) => v != null);
  if (!labels.length) {
    legendEl.innerHTML = "";
    return;
  }
  const counts = new Map();
  labels.forEach((l) => counts.set(l, (counts.get(l) ?? 0) + 1));
  const html = [...counts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([label, count]) => {
      const color = label === -1 ? "#9aa0ab" : palette[label % palette.length];
      const title = label === -1 ? "噪声" : `簇 ${label}`;
      return `<span class="legend-item"><span class="legend-dot" style="background:${color};"></span>${title}: ${count}</span>`;
    })
    .join("");
  legendEl.innerHTML = `${html}<span class="legend-item"><span class="legend-dot" style="background:#111;"></span>中心标记</span>`;
}

function clearAnimation() {
  if (state.animation.timerId) {
    clearInterval(state.animation.timerId);
  }
  state.animation = {
    isAnimating: false,
    frames: [],
    frameIndex: 0,
    timerId: null,
    activeCentroids: null,
  };
}

function updateAnimationButtons() {
  const hasFrames = state.animation.frames.length > 0;
  controls.playBtn.disabled = !hasFrames || state.animation.isAnimating;
  controls.pauseBtn.disabled = !state.animation.isAnimating;
  controls.stepBtn.disabled = !hasFrames;
}

function applyLabels(labels) {
  state.dataset.forEach((p, i) => {
    const next = labels[i];
    p.predCluster = next == null ? null : next;
  });
}

function applyFrame(frame) {
  applyLabels(frame.labels ?? []);
  state.animation.activeCentroids = frame.centroids ?? null;
}

function applyFinalResult() {
  if (!state.clusterResult) {
    return;
  }
  applyLabels(state.clusterResult.labels);
  state.animation.activeCentroids = state.currentAlgorithm === "kmeans" ? state.clusterResult.centroids : null;
}

function playAnimation() {
  if (!state.animation.frames.length) {
    return;
  }
  if (state.animation.frameIndex >= state.animation.frames.length) {
    state.animation.frameIndex = 0;
  }
  state.animation.isAnimating = true;
  updateAnimationButtons();
  const speed = Number(controls.speedSelect.value) || 1;
  const interval = Math.max(80, Math.floor(420 / speed));
  state.animation.timerId = setInterval(() => {
    const frame = state.animation.frames[state.animation.frameIndex];
    if (!frame) {
      pauseAnimation(true);
      return;
    }
    applyFrame(frame);
    state.animation.frameIndex += 1;
    if (state.animation.frameIndex >= state.animation.frames.length) {
      pauseAnimation(true);
      applyFinalResult();
      setStatus("动画播放完成。");
    }
    renderAll();
  }, interval);
}

function pauseAnimation(finished = false) {
  if (state.animation.timerId) {
    clearInterval(state.animation.timerId);
  }
  state.animation.timerId = null;
  state.animation.isAnimating = false;
  if (!finished) {
    setStatus("动画已暂停。");
  }
  updateAnimationButtons();
}

function stepAnimation() {
  if (!state.animation.frames.length) {
    return;
  }
  if (state.animation.frameIndex >= state.animation.frames.length) {
    state.animation.frameIndex = 0;
  }
  const frame = state.animation.frames[state.animation.frameIndex];
  applyFrame(frame);
  state.animation.frameIndex += 1;
  renderAll();
}

function runClustering() {
  if (!state.dataset.length) {
    return;
  }
  state.params = readParams();
  state.currentAlgorithm = controls.algorithm.value;
  state.selectedPointId = null;
  state.pointExplanation = null;
  clearAnimation();
  const purePoints = state.dataset.map((p) => ({ x: p.x, y: p.y }));

  if (state.currentAlgorithm === "kmeans") {
    const result = runKMeans(purePoints, {
      k: state.params.kmeansK,
      maxIter: state.params.kmeansMaxIter,
      tol: state.params.kmeansTol,
    });
    state.clusterResult = result;
    state.dataset.forEach((p, i) => {
      p.predCluster = result.labels[i];
    });
    state.animation.frames = result.frames ?? [];
  } else if (state.currentAlgorithm === "dbscan") {
    const result = runDBSCAN(purePoints, {
      eps: state.params.dbscanEps,
      minPts: state.params.dbscanMinPts,
    });
    state.clusterResult = result;
    state.dataset.forEach((p, i) => {
      p.predCluster = result.labels[i];
    });
    state.animation.frames = result.frames ?? [];
  } else {
    const result = runAgglomerative(purePoints, { k: state.params.aggloK });
    state.clusterResult = result;
    state.dataset.forEach((p, i) => {
      p.predCluster = result.labels[i];
    });
    state.animation.frames = result.frames ?? [];
  }
  state.animation.frameIndex = 0;
  updateAnimationButtons();
  setStatus(`已完成 ${state.currentAlgorithm} 聚类，可点击点查看解释或播放过程动画。`);
  renderAll();
}

function findNearestPoint(screenX, screenY) {
  let nearest = null;
  let best = Infinity;
  state.dataset.forEach((point) => {
    const p = worldToScreen(point.x, point.y);
    const dx = p.x - screenX;
    const dy = p.y - screenY;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d < best) {
      best = d;
      nearest = point;
    }
  });
  return best <= 12 ? nearest : null;
}

function onCanvasClick(ev) {
  if (!state.dataset.length) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const x = ((ev.clientX - rect.left) * canvas.width) / rect.width;
  const y = ((ev.clientY - rect.top) * canvas.height) / rect.height;
  const nearest = findNearestPoint(x, y);
  if (!nearest) {
    return;
  }

  state.selectedPointId = nearest.id;
  state.pointExplanation = null;
  if (state.clusterResult) {
    const index = state.dataset.findIndex((p) => p.id === nearest.id);
    if (state.currentAlgorithm === "kmeans") {
      state.pointExplanation = explainPointKMeans(nearest, state.clusterResult.centroids);
    } else if (state.currentAlgorithm === "dbscan") {
      state.pointExplanation = explainPointDBSCAN(index, state.clusterResult);
    } else {
      state.pointExplanation = explainPointAgglomerative(index, state.clusterResult);
    }
  }
  renderAll();
}

function syncAlgorithmParamVisibility() {
  const algo = controls.algorithm.value;
  controls.kmeansParams.classList.toggle("hidden", algo !== "kmeans");
  controls.dbscanParams.classList.toggle("hidden", algo !== "dbscan");
  controls.aggloParams.classList.toggle("hidden", algo !== "agglomerative");
}

function invalidateClusterResult() {
  clearAnimation();
  state.clusterResult = null;
  state.selectedPointId = null;
  state.pointExplanation = null;
  state.dataset.forEach((p) => {
    p.predCluster = null;
  });
  updateAnimationButtons();
  setStatus("参数已变更，请重新执行聚类。");
  renderAll();
}

function resetAll() {
  setParams(DEFAULT_PARAMS);
  state.currentAlgorithm = "kmeans";
  controls.algorithm.value = "kmeans";
  state.dataset = [];
  state.clusterResult = null;
  state.selectedPointId = null;
  state.pointExplanation = null;
  state.bounds = { minX: -60, maxX: 60, minY: -60, maxY: 60 };
  clearAnimation();
  controls.clusterBtn.disabled = true;
  syncAlgorithmParamVisibility();
  updateAnimationButtons();
  setStatus("已重置为默认参数。");
  renderAll();
}

function randomInt(min, max, rng) {
  return Math.floor(rng() * (max - min + 1)) + min;
}

function randomizeParams() {
  const rng = mulberry32(Date.now() % 1000000);
  const numClusters = randomInt(2, 6, rng);
  const pointsPerCluster = randomInt(40, 150, rng);
  const stdDev = randomInt(2, 10, rng);
  const spread = randomInt(25, 60, rng);
  const kmeansK = Math.max(2, Math.min(10, numClusters + randomInt(-1, 1, rng)));
  const dbscanEps = Number((stdDev * (0.9 + rng())).toFixed(1));
  const dbscanMinPts = randomInt(4, 10, rng);
  const seed = randomInt(1, 9999999, rng);
  setParams({
    numClusters,
    pointsPerCluster,
    stdDev,
    spread,
    seed,
    kmeansK,
    kmeansMaxIter: 50,
    kmeansTol: 0.001,
    dbscanEps,
    dbscanMinPts,
    aggloK: numClusters,
  });
  invalidateClusterResult();
  setStatus("已随机生成参数，请先生成数据再聚类。");
}

function renderAll() {
  drawAxesAndGrid();
  drawPoints();
  drawClusterCenters();
  renderLegend();
  renderAlgorithmSummary(state, summaryEl);
  renderPointDetail(state, pointEl);
}

controls.generateBtn.addEventListener("click", generateDataset);
controls.randomParamsBtn.addEventListener("click", randomizeParams);
controls.resetBtn.addEventListener("click", resetAll);
controls.clusterBtn.addEventListener("click", runClustering);
controls.playBtn.addEventListener("click", playAnimation);
controls.pauseBtn.addEventListener("click", () => pauseAnimation(false));
controls.stepBtn.addEventListener("click", stepAnimation);
controls.algorithm.addEventListener("change", () => {
  state.currentAlgorithm = controls.algorithm.value;
  syncAlgorithmParamVisibility();
  invalidateClusterResult();
});

[
  controls.kmeansK,
  controls.kmeansMaxIter,
  controls.kmeansTol,
  controls.dbscanEps,
  controls.dbscanMinPts,
  controls.aggloK,
].forEach((el) => {
  el.addEventListener("change", () => {
    if (state.clusterResult) {
      invalidateClusterResult();
    }
  });
});

controls.numClusters.addEventListener("change", () => {
  controls.aggloK.value = controls.numClusters.value;
  if (state.clusterResult) {
    invalidateClusterResult();
  }
});

[
  controls.pointsPerCluster,
  controls.stdDev,
  controls.spread,
  controls.seed,
].forEach((el) => {
  el.addEventListener("change", () => {
    if (state.clusterResult) {
      invalidateClusterResult();
    }
  });
});

canvas.addEventListener("click", onCanvasClick);

setParams(DEFAULT_PARAMS);
syncAlgorithmParamVisibility();
updateAnimationButtons();
setStatus("准备就绪。");
renderAll();
