function sqDist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function nearestCentroid(point, centroids) {
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < centroids.length; i += 1) {
    const d = sqDist(point, centroids[i]);
    if (d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  }
  return { index: bestIdx, distance: bestDist };
}

export function runKMeans(points, options = {}) {
  const k = Math.max(1, Math.floor(options.k ?? 3));
  const maxIter = Math.max(1, Math.floor(options.maxIter ?? 50));
  const tol = Math.max(0, Number(options.tol ?? 1e-3));
  if (!points.length) {
    return { labels: [], centroids: [], logs: [], iterations: 0 };
  }

  const centroids = [];
  for (let i = 0; i < k; i += 1) {
    const src = points[i % points.length];
    centroids.push({ x: src.x, y: src.y });
  }

  const labels = new Array(points.length).fill(0);
  const logs = [];
  const frames = [];
  let iterations = 0;

  for (let iter = 0; iter < maxIter; iter += 1) {
    iterations = iter + 1;

    for (let i = 0; i < points.length; i += 1) {
      labels[i] = nearestCentroid(points[i], centroids).index;
    }
    frames.push({
      stage: "assign",
      iter: iter + 1,
      labels: [...labels],
      centroids: centroids.map((c) => ({ x: c.x, y: c.y })),
    });

    const sums = Array.from({ length: k }, () => ({ x: 0, y: 0, c: 0 }));
    for (let i = 0; i < points.length; i += 1) {
      const cls = labels[i];
      sums[cls].x += points[i].x;
      sums[cls].y += points[i].y;
      sums[cls].c += 1;
    }

    let maxShift = 0;
    for (let j = 0; j < k; j += 1) {
      if (sums[j].c === 0) {
        continue;
      }
      const next = { x: sums[j].x / sums[j].c, y: sums[j].y / sums[j].c };
      const shift = Math.sqrt(sqDist(next, centroids[j]));
      maxShift = Math.max(maxShift, shift);
      centroids[j] = next;
    }

    logs.push({
      iter: iter + 1,
      maxShift,
      centroids: centroids.map((c) => ({ x: c.x, y: c.y })),
    });
    frames.push({
      stage: "update",
      iter: iter + 1,
      labels: [...labels],
      centroids: centroids.map((c) => ({ x: c.x, y: c.y })),
      maxShift,
    });

    if (maxShift < tol) {
      break;
    }
  }

  return { labels, centroids, logs, iterations, frames };
}

export function explainPointKMeans(point, centroids) {
  const distances = centroids.map((c, idx) => {
    const d = Math.sqrt(sqDist(point, c));
    return { cluster: idx, distance: d, centroid: c };
  });
  distances.sort((a, b) => a.distance - b.distance);
  const winner = distances[0];

  return {
    point,
    winnerCluster: winner.cluster,
    distances,
  };
}
