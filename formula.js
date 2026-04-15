function fixed(v) {
  return Number(v).toFixed(3);
}

function bar(value, max, color = "#5a88ec") {
  const pct = max <= 0 ? 0 : Math.max(2, Math.min(100, (value / max) * 100));
  return `<div class="bar-wrap"><div class="bar-fill" style="width:${pct}%;background:${color};"></div></div>`;
}

export function renderAlgorithmSummary(state, summaryEl) {
  const { currentAlgorithm, clusterResult, dataset, params } = state;
  if (!clusterResult) {
    summaryEl.textContent = "请先生成数据并执行聚类。";
    return;
  }

  if (currentAlgorithm === "kmeans") {
    const sizes = countLabels(clusterResult.labels);
    summaryEl.innerHTML = `
<div class="section-title">算法: K-Means</div>
<div class="math-block"><strong>输入</strong> 点集=${dataset.length}, K=${params.kmeansK}, maxIter=${params.kmeansMaxIter}, tol=${params.kmeansTol}</div>
<div class="math-block">
  <strong>计算过程</strong>
  <div class="equation">分配: argmin_j ||x_i - μ_j||^2</div>
  <div class="equation">更新: μ_j = mean(C_j)</div>
  <div class="math-step">迭代轮数: ${clusterResult.iterations}</div>
</div>
<div class="math-block"><strong>输出</strong> 簇规模=${sizes}</div>`;
    return;
  }

  if (currentAlgorithm === "dbscan") {
    const sizes = countLabels(clusterResult.labels);
    summaryEl.innerHTML = `
<div class="section-title">算法: DBSCAN</div>
<div class="math-block"><strong>输入</strong> 点集=${dataset.length}, eps=${params.dbscanEps}, minPts=${params.dbscanMinPts}</div>
<div class="math-block">
  <strong>计算过程</strong>
  <div class="equation">邻域: N_eps(p) = { q | dist(p,q) <= eps }</div>
  <div class="equation">判定: |N_eps(p)| >= minPts => 核心点</div>
</div>
<div class="math-block"><strong>输出</strong> 簇数量=${clusterResult.clusterCount}, 簇规模=${sizes}</div>`;
    return;
  }

  const sizes = countLabels(clusterResult.labels);
  summaryEl.innerHTML = `
<div class="section-title">算法: Agglomerative (Average Linkage)</div>
<div class="math-block"><strong>输入</strong> 点集=${dataset.length}, K=${params.aggloK}</div>
<div class="math-block">
  <strong>计算过程</strong>
  <div class="equation">d_avg(A,B)= (1/(|A||B|)) * Σ dist(a,b)</div>
  <div class="equation">重复合并 d_avg 最小的两个簇，直到簇数=K</div>
</div>
<div class="math-block"><strong>输出</strong> 最终簇数量=${clusterResult.clusterCount}, 簇规模=${sizes}</div>`;
}

export function renderPointDetail(state, pointEl) {
  const { selectedPointId, dataset, currentAlgorithm, clusterResult, pointExplanation } = state;
  if (selectedPointId == null) {
    pointEl.textContent = "在坐标系中点击一个点，查看该点的输入/计算过程/输出。";
    return;
  }

  const point = dataset.find((p) => p.id === selectedPointId);
  if (!point) {
    pointEl.textContent = "未找到选中点。";
    return;
  }

  if (!clusterResult || !pointExplanation) {
    pointEl.textContent = `点输入: x=(${fixed(point.x)}, ${fixed(point.y)})\n当前尚未执行聚类。`;
    return;
  }

  if (currentAlgorithm === "kmeans") {
    const maxDist = Math.max(...pointExplanation.distances.map((d) => d.distance), 1);
    const compareRows = pointExplanation.distances
      .map((d) => {
        const txt = `μ_${d.cluster}: ${fixed(d.distance)}`;
        return `<div class="compare-row">${txt}${bar(d.distance, maxDist)}</div>`;
      })
      .join("");
    pointEl.innerHTML = `
<div class="section-title">选中点推导</div>
<div class="math-block"><strong>输入</strong> x=(${fixed(point.x)}, ${fixed(point.y)})</div>
<div class="math-block"><strong>计算过程</strong><div class="equation">dist(x, μ_j) 对比</div>${compareRows}</div>
<div class="math-block"><strong>输出</strong> argmin 对应簇 = ${pointExplanation.winnerCluster}</div>`;
    return;
  }

  if (currentAlgorithm === "dbscan") {
    const compare = bar(pointExplanation.neighbors.length, Math.max(pointExplanation.minPts, pointExplanation.neighbors.length));
    pointEl.innerHTML = `
<div class="section-title">选中点推导</div>
<div class="math-block"><strong>输入</strong> x=(${fixed(point.x)}, ${fixed(point.y)})</div>
<div class="math-block"><strong>计算过程</strong> |N_eps(p)|=${pointExplanation.neighbors.length}, minPts=${pointExplanation.minPts}${compare}</div>
<div class="math-block"><strong>输出</strong> 点类型=${pointExplanation.pointType}, 所属簇=${pointExplanation.label}</div>`;
    return;
  }

  const mergeText = pointExplanation.latestMerge
    ? `最近相关合并: 第${pointExplanation.latestMerge.step}步，合并簇 ${pointExplanation.latestMerge.mergeA} 与 ${pointExplanation.latestMerge.mergeB}，距离=${fixed(
        pointExplanation.latestMerge.distance
      )}`
    : "该点在最终阶段未记录到相关合并信息。";
  const clusterSizeMax = Math.max(...clusterResult.labels.map((label) => clusterResult.labels.filter((x) => x === label).length), 1);
  pointEl.innerHTML = `
<div class="section-title">选中点推导</div>
<div class="math-block"><strong>输入</strong> x=(${fixed(point.x)}, ${fixed(point.y)})</div>
<div class="math-block"><strong>计算过程</strong> ${mergeText}</div>
<div class="math-block"><strong>局部对比</strong> 簇大小=${pointExplanation.clusterSize}${bar(pointExplanation.clusterSize, clusterSizeMax, "#6e9c56")}</div>
<div class="math-block"><strong>输出</strong> 所属簇=${pointExplanation.label}, 该簇大小=${pointExplanation.clusterSize}</div>`;
}

function countLabels(labels) {
  const map = new Map();
  labels.forEach((l) => map.set(l, (map.get(l) ?? 0) + 1));
  return [...map.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([label, count]) => `${label}:${count}`)
    .join(", ");
}
