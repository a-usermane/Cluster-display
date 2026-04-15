function euclidean(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function averageLinkageDistance(clusterA, clusterB, points) {
  let total = 0;
  let count = 0;
  for (let i = 0; i < clusterA.members.length; i += 1) {
    for (let j = 0; j < clusterB.members.length; j += 1) {
      total += euclidean(points[clusterA.members[i]], points[clusterB.members[j]]);
      count += 1;
    }
  }
  return count ? total / count : Infinity;
}

export function runAgglomerative(points, options = {}) {
  if (!points.length) {
    return { labels: [], clusterCount: 0, logs: [], pointClusterMap: [] };
  }

  const targetK = Math.max(1, Math.min(Math.floor(options.k ?? 2), points.length));
  let nextClusterId = points.length;
  const logs = [];
  const frames = [];

  let clusters = points.map((_, idx) => ({
    id: idx,
    members: [idx],
  }));

  while (clusters.length > targetK) {
    let bestI = 0;
    let bestJ = 1;
    let bestDistance = Infinity;

    for (let i = 0; i < clusters.length; i += 1) {
      for (let j = i + 1; j < clusters.length; j += 1) {
        const d = averageLinkageDistance(clusters[i], clusters[j], points);
        if (d < bestDistance) {
          bestDistance = d;
          bestI = i;
          bestJ = j;
        }
      }
    }

    const a = clusters[bestI];
    const b = clusters[bestJ];
    const merged = {
      id: nextClusterId++,
      members: [...a.members, ...b.members],
    };

    logs.push({
      step: logs.length + 1,
      mergeA: a.id,
      mergeB: b.id,
      mergedId: merged.id,
      distance: bestDistance,
      sizeAfterMerge: merged.members.length,
      remainClusters: clusters.length - 1,
    });

    clusters = clusters.filter((_, idx) => idx !== bestI && idx !== bestJ);
    clusters.push(merged);

    const frameLabels = new Array(points.length).fill(-1);
    clusters.forEach((cluster, clusterIdx) => {
      cluster.members.forEach((pointIdx) => {
        frameLabels[pointIdx] = clusterIdx;
      });
    });
    frames.push({
      stage: "merge",
      step: logs.length,
      labels: frameLabels,
      mergedId: merged.id,
      distance: bestDistance,
    });
  }

  clusters.sort((x, y) => x.id - y.id);
  const labels = new Array(points.length).fill(-1);
  const pointClusterMap = new Array(points.length).fill(-1);
  clusters.forEach((cluster, labelIdx) => {
    cluster.members.forEach((pointIdx) => {
      labels[pointIdx] = labelIdx;
      pointClusterMap[pointIdx] = cluster.id;
    });
  });

  return {
    labels,
    clusterCount: clusters.length,
    logs,
    finalClusters: clusters.map((c, idx) => ({
      label: idx,
      nodeId: c.id,
      members: [...c.members],
    })),
    pointClusterMap,
    frames,
  };
}

export function explainPointAgglomerative(pointIndex, result) {
  const label = result.labels[pointIndex];
  const finalCluster = result.finalClusters.find((c) => c.label === label);
  const relatedMerges = result.logs.filter(
    (item) => item.mergedId === result.pointClusterMap[pointIndex] || item.mergeA === result.pointClusterMap[pointIndex] || item.mergeB === result.pointClusterMap[pointIndex]
  );
  const latestMerge = relatedMerges.length ? relatedMerges[relatedMerges.length - 1] : null;

  return {
    pointIndex,
    label,
    clusterSize: finalCluster ? finalCluster.members.length : 0,
    latestMerge,
  };
}
