// Option A: with an --external-experts helper card beside the engine's, the Monitor gets one row per card under the
// eight tiles (which show the cards together): its role, load, VRAM, temperature, power and its own PCIe link.
// Its own file and its own /metrics poll, so app.js stays as upstream has it; index.html only loads it.
"use strict";
(() => {
  const view = document.getElementById("view-monitor");
  const tiles = document.getElementById("metrics");
  if (!view || !tiles) return;

  const card = document.createElement("div");
  card.className = "st-card gpu-cards";
  card.hidden = true;
  card.innerHTML = `
    <div class="req-head"><span class="card-title">Graphics cards</span><span class="muted small">each card on its own</span></div>
    <div class="table-wrap"><table class="st-table">
      <thead><tr><th>Card</th><th>Role</th><th class="num">Load</th><th>VRAM</th><th class="num">Temp</th>
        <th class="num">Power</th><th>PCIe link</th><th class="num">PCIe to / from card</th></tr></thead>
      <tbody></tbody></table></div>`;
  tiles.after(card);
  const style = document.createElement("style");
  style.textContent = `
    .gpu-cards { display: flex; flex-direction: column; gap: var(--st-s-3); }
    .gpu-cards td { white-space: nowrap; }
    .gpu-cards .vram { display: flex; align-items: center; gap: var(--st-s-2); min-width: 160px; }
    .gpu-cards .vram .st-progress { flex: 1; min-width: 60px; }`;
  document.head.append(style);
  const body = card.querySelector("tbody");

  const num = (v, d = 0) => (v == null || Number.isNaN(Number(v)) ? "–"
    : Number(v).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
  const gib = (b) => num(b == null ? null : b / 1073741824, 1);   // binary GB, as the tiles show it
  const mbs = (v) => (v == null ? "–" : `${num(v, v < 10 ? 1 : 0)} MB/s`);
  function key() {
    try { const v = localStorage.getItem("strata.apikey"); return v === null ? "" : JSON.parse(v); } catch (e) { return ""; }
  }
  function cell(row, text, cls) {
    const td = row.insertCell();
    if (cls) td.className = cls;
    td.textContent = text;
    return td;
  }

  function render(gpus) {
    // only beside a helper card: without one, the tiles' per-card lines stay the Monitor's only per-card view
    card.hidden = !gpus.some((g) => g.helper);
    if (card.hidden) return;
    body.replaceChildren();
    for (const g of gpus) {
      const row = body.insertRow();
      const name = cell(row, g.name || (g.helper ? "AMD card" : `GPU ${g.index}`));
      name.title = `device ${g.index}`;
      cell(row, g.helper ? "Expert helper" : "Engine", "muted");
      cell(row, g.util == null ? "–" : `${num(g.util)}%`, "num");
      const vram = row.insertCell().appendChild(document.createElement("div"));
      vram.className = "vram";
      const bar = vram.appendChild(document.createElement("div"));
      bar.className = "st-progress";
      const fill = bar.appendChild(document.createElement("div"));
      fill.className = "st-progress__bar";
      const frac = g.mem_used != null && g.mem_total ? g.mem_used / g.mem_total : 0;
      fill.style.width = `${Math.min(100, 100 * frac)}%`;   // full is normal: the expert caches fill the cards
      vram.appendChild(document.createElement("span")).textContent = `${gib(g.mem_used)} / ${gib(g.mem_total)} GB`;
      cell(row, g.temp == null ? "–" : `${num(g.temp)} °C`, "num");
      cell(row, g.power == null ? "–" : `${num(g.power)}${g.power_limit ? ` / ${num(g.power_limit)}` : ""} W`, "num");
      const gen = g.pcie_gen_max || g.pcie_gen;
      cell(row, gen ? `Gen${gen}${g.pcie_width ? ` x${g.pcie_width}` : ""}` +
                      (g.pcie_gen && g.pcie_gen < gen ? ` (idle Gen${g.pcie_gen})` : "") : "–");
      cell(row, g.pcie_rx_mb == null && g.pcie_tx_mb == null ? "–" : `${mbs(g.pcie_rx_mb)} / ${mbs(g.pcie_tx_mb)}`, "num");
    }
  }

  async function poll() {
    if (!view.hidden && document.visibilityState === "visible") {
      try {
        const k = key();
        const r = await fetch("metrics", {headers: k ? {Authorization: "Bearer " + k} : {}});
        if (r.ok) render(((await r.json()).hardware || {}).gpus || []);
      } catch (e) { /* app.js reports an unreachable server */ }
    }
    setTimeout(poll, 1000);
  }
  poll();
})();
