function euclidean(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function regionQuery(points, idx, eps) {
  const neighbors = [];
  for (let i = 0; i < points.length; i += 1) {
    if (euclidean(points[idx], points[i]) <= eps) {
      neighbors.push(i);
    }
  }
  return neighbors;
}

export function runDBSCAN(points, options = {}) {
  const eps = Math.max(0.1, Number(options.eps ?? 8));
  const minPts = Math.max(2, Math.floor(options.minPts ?? 6));
  const labels = new Array(points.length).fill(undefined);
  const visited = new Array(points.length).fill(false);
  const neighborMap = new Map();
  const logs = [];
  const frames = [];
  const corePointSet = new Set();
  let clusterId = 0;

  for (let i = 0; i < points.length; i += 1) {
    const n = regionQuery(points, i, eps);
    neighborMap.set(i, n);
    if (n.length >= minPts) {
      corePointSet.add(i);
    }
  }

  for (let i = 0; i < points.length; i += 1) {
    if (visited[i]) {
      continue;
    }
    visited[i] = true;
    let neighbors = neighborMap.get(i);

    if (neighbors.length < minPts) {
      labels[i] = -1;
      logs.push({ type: "noise", pointIndex: i, neighbors: neighbors.length });
      frames.push({
        stage: "noise",
        pointIndex: i,
        labels: labels.map((v) => (v === undefined ? null : v)),
      });
      continue;
    }

    labels[i] = clusterId;
    const queue = [...neighbors];
    logs.push({ type: "startCluster", clusterId, seed: i, neighborCount: neighbors.length });
    frames.push({
      stage: "startCluster",
      clusterId,
      seed: i,
      labels: labels.map((v) => (v === undefined ? null : v)),
    });

    while (queue.length) {
      const current = queue.shift();
      if (!visited[current]) {
        visited[current] = true;
        const currentNeighbors = neighborMap.get(current);
        if (currentNeighbors.length >= minPts) {
          queue.push(...currentNeighbors);
        }
      }
      if (labels[current] === undefined || labels[current] === -1) {
        labels[current] = clusterId;
        frames.push({
          stage: "expand",
          clusterId,
          pointIndex: current,
          labels: labels.map((v) => (v === undefined ? null : v)),
        });
      }
    }

    clusterId += 1;
  }

  return {
    labels,
    logs,
    clusterCount: clusterId,
    eps,
    minPts,
    neighborMap,
    corePointSet,
    frames,
  };
}

export function explainPointDBSCAN(pointIndex, result) {
  const neighbors = result.neighborMap.get(pointIndex) ?? [];
  const label = result.labels[pointIndex];
  const isCore = result.corePointSet.has(pointIndex);
  const pointType = label === -1 ? "噪声点" : isCore ? "核心点" : "边界点";

  return {
    pointIndex,
    label,
    pointType,
    neighbors,
    eps: result.eps,
    minPts: result.minPts,
  };
}
