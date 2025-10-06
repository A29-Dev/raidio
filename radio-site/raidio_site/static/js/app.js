(function () {
  const WS_URL = window.RADIO.WS_URL || "ws://127.0.0.1:8000/live";
  const DEFAULT_FOCUS = (window.RADIO.DEFAULT_FOCUS || []).map(x => x.toLowerCase());

  const transcriptEl = document.getElementById("transcript");
  const heatmapEl = document.getElementById("heatmap");
  const entityChipsEl = document.getElementById("entityChips");
  const wikiLinksEl = document.getElementById("wikiLinks");
  const connBadge = document.getElementById("connBadge");
  const clockEl = document.getElementById("clock");
  const uptimeEl = document.getElementById("uptime");
  const pingEl = document.getElementById("ping");

  // clock/uptime
  const startTs = Date.now();
  setInterval(() => {
    const now = new Date();
    clockEl.textContent = now.toLocaleString();
    const s = Math.floor((Date.now() - startTs) / 1000);
    const h = String(Math.floor(s / 3600)).padStart(2, "0");
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    uptimeEl.textContent = `⏱ ${h}:${m}:${ss}`;
  }, 500);

  // poor-man's TCP connect latency
  async function measurePing() {
    try {
      if (WS_URL.startsWith("ws")) {
        const u = new URL(WS_URL);
        const test = (u.protocol === "wss:") ? `https://${u.host}/` : `http://${u.host}/`;
        const t0 = performance.now();
        await fetch(test, { mode: "no-cors", cache: "no-store" });
        pingEl.textContent = `📶 ${Math.round(performance.now() - t0)} ms`;
      }
    } catch { pingEl.textContent = "📶 —"; }
  }
  setInterval(measurePing, 10000); measurePing();

  // Leaflet map
  const map = L.map(document.getElementById("map")).setView([-31.95, 115.86], 10);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "© OpenStreetMap" }).addTo(map);
  let markerLayer = L.layerGroup().addTo(map);

  // state
  const events = [];
  const freq = {}; // keyword frequencies

  function tokenize(t) {
    return (t || "").toLowerCase().replace(/[^a-z0-9\s']/gi, " ").split(/\s+/).filter(Boolean);
  }
  function highlight(text, terms) {
    if (!terms.length) return text;
    const esc = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const re = new RegExp(`(\\b(?:${esc.join("|")})\\b)`, "gi");
    const out = [];
    let last = 0, m;
    while ((m = re.exec(text))) {
      const start = m.index, end = start + m[0].length;
      if (start > last) out.push(text.slice(last, start));
      const mark = document.createElement("mark"); mark.textContent = text.slice(start, end);
      out.push(mark.outerHTML); last = end;
    }
    if (last < text.length) out.push(text.slice(last));
    return out.join("");
  }
  function renderHeatmap(keys) {
    heatmapEl.innerHTML = "";
    if (!keys.length) return;
    const max = Math.max(1, ...keys.map(k => freq[k] || 0));
    keys.forEach(k => {
      const n = freq[k] || 0;
      const span = document.createElement("span");
      span.className = "chip";
      span.style.fontSize = (12 + (n / max) * 22) + "px";
      span.style.opacity = (0.35 + (n / max) * 0.65);
      span.title = `${k}: ${n}`;
      span.textContent = k;
      heatmapEl.appendChild(span);
    });
  }

  // WebSocket
  const ws = new WebSocket(WS_URL);
  ws.onopen = () => { connBadge.textContent = "Connected"; connBadge.classList.remove("off"); };
  ws.onclose = () => { connBadge.textContent = "Disconnected"; connBadge.classList.add("off"); };
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      events.push(data);
      // focus words: cartridge awareness or default
      const awareKW = data?.awareness?.keywords ? Object.keys(data.awareness.keywords) : [];
      const focus = (awareKW.length ? awareKW : DEFAULT_FOCUS).map(x => x.toLowerCase());

      // update freq for focus words only
      if (data?.text) {
        const toks = tokenize(data.text);
        toks.forEach(t => { if (focus.includes(t)) freq[t] = (freq[t] || 0) + 1; });
        renderHeatmap(Array.from(new Set([...focus, ...Object.keys(freq)])));
      }

      // transcript row (no truncation)
      const row = document.createElement("div"); row.className = "row";
      const tEl = document.createElement("div"); tEl.className = "time"; tEl.textContent = new Date(data.t0 || Date.now()).toLocaleTimeString();
      const xEl = document.createElement("div"); xEl.className = "txt"; xEl.innerHTML = highlight(data.text || "", Array.from(new Set([...focus, ...(data.entities||[]).map(e => (e.text || '').toLowerCase())])));
      row.appendChild(tEl); row.appendChild(xEl);
      transcriptEl.prepend(row);

      // entities as chips (click to add highlight)
      entityChipsEl.innerHTML = "";
      (data.entities || []).slice(0, 12).forEach(e => {
        const b = document.createElement("button");
        b.className = "chip"; b.textContent = `${e.type}: ${e.text}`;
        b.onclick = () => {
          const merged = Array.from(new Set([ ...(focus || []), (e.text || "").toLowerCase() ]));
          xEl.innerHTML = highlight(data.text || "", merged);
        };
        entityChipsEl.appendChild(b);
      });

      // map markers from entities with lat/lon
      markerLayer.clearLayers();
      const pts = (data.entities || []).filter(e => typeof e.lat === "number" && typeof e.lon === "number");
      pts.forEach(p => L.marker([p.lat, p.lon]).addTo(markerLayer).bindPopup(`<b>${p.text}</b><br/>${data.text || ""}`));
      if (pts.length) {
        const bounds = L.latLngBounds(pts.map(p => [p.lat, p.lon]));
        map.fitBounds(bounds.pad(0.25));
      }

      // wiki links
      wikiLinksEl.innerHTML = "";
      const wikiTerms = Array.from(new Set((data.entities||[])
        .filter(e => ["PERSON","ORG","GPE","LOC","FAC"].includes(e.type))
        .map(e => e.text))).slice(0, 8);
      wikiTerms.forEach(t => {
        const a = document.createElement("a");
        a.href = `https://en.wikipedia.org/wiki/Special:Search?search=${encodeURIComponent(t)}`;
        a.target = "_blank"; a.className = "chip"; a.textContent = t;
        wikiLinksEl.appendChild(a);
      });
    } catch {}
  };
})();
