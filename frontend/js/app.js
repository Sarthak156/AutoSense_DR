/* ============================================================
   AutoSense — frontend application
   Telemetry: WebSocket `/ws` with HTTP polling fallback.
   Video: MJPEG stream (snapshot polling optional).
   All values rendered come from the live perception backend.
   ============================================================ */
(() => {
"use strict";

/* ---------------- helpers ---------------- */
const $  = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const fmt0 = v => (v === null || v === undefined) ? "—" : Math.round(v);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const ago = ts => { const s = Math.max(0, Date.now()/1000 - ts); return s < 2 ? "now" : s < 60 ? `${s|0}s ago` : `${s/60|0}m ago`; };
/* byte-sequence search inside a Uint8Array (used by the MJPEG demuxer) */
function findSeq(hay, needle, from = 0) {
  outer: for (let i = from; i <= hay.length - needle.length; i++) {
    if (hay[i] !== needle[0]) continue;
    for (let j = 1; j < needle.length; j++) if (hay[i + j] !== needle[j]) continue outer;
    return i;
  }
  return -1;
}

async function api(path, method = "GET", body) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  if (!r.ok) { let e; try { e = (await r.json()).error; } catch {} throw new Error(e || `${r.status}`); }
  return r.json();
}

function flattenConfigPatch(patch, prefix = "", out = []) {
  for (const [key, value] of Object.entries(patch || {})) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value)) flattenConfigPatch(value, path, out);
    else out.push([path, value]);
  }
  return out;
}
function getConfigPath(config, path) {
  return path.split(".").reduce((value, key) => value?.[key], config);
}
function setConfigPath(config, path, value) {
  const parts = path.split(".");
  let target = config;
  for (const key of parts.slice(0, -1)) target = target[key] ||= {};
  target[parts[parts.length - 1]] = value;
}
function sameConfigValue(a, b) {
  return typeof a === "number" && typeof b === "number" ? Math.abs(a - b) < 1e-9 : a === b;
}

const postCfg = (() => {  // debounced config push with nested ROI patch merging
  let t = null, pending = {}, revision = 0;
  const merge = (dst, src) => {
    for (const [key, value] of Object.entries(src)) {
      if (value && typeof value === "object" && !Array.isArray(value)) {
        dst[key] = merge(dst[key] && typeof dst[key] === "object" ? dst[key] : {}, value);
      } else dst[key] = value;
    }
    return dst;
  };
  return patch => {
    merge(pending, patch);
    const thisRevision = ++revision;
    for (const [path, value] of flattenConfigPatch(patch)) {
      State.configOverrides.set(path, { value, revision: thisRevision });
    }
    clearTimeout(t);
    t = setTimeout(() => {
      const p = pending;
      pending = {};
      const sent = flattenConfigPatch(p).map(([path, value]) => ({
        path, value, revision: State.configOverrides.get(path)?.revision,
      }));
      api("/api/config", "POST", p).then(result => {
        // Keep acknowledged edits visible until telemetry echoes the new config;
        // otherwise a lagging stats frame can snap the sliders back.
        for (const item of sent) {
          const current = State.configOverrides.get(item.path);
          if (!current || current.revision !== item.revision) continue;
          const actual = getConfigPath(result.config, item.path);
          if (actual !== undefined) current.value = actual;
        }
      }).catch(error => {
        for (const item of sent) {
          if (State.configOverrides.get(item.path)?.revision === item.revision) {
            State.configOverrides.delete(item.path);
          }
        }
        toast(`Setting was not applied: ${error.message}`, "err");
        if (State.stats?.config) syncControls(State.stats.config);
      });
    }, 160);
  };
})();

function toast(msg, kind = "") {
  const box = $("#toasts");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<span class="dot ${kind === "ok" ? "green" : kind === "err" ? "red" : "gray"}"></span>${msg}`;
  box.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .4s"; setTimeout(() => el.remove(), 420); }, 4200);
}

/* ---------------- state ---------------- */
const State = {
  stats: null,
  lastUpdate: 0,
  transport: "—",
  ws: null,
  pollTimer: null,
  editing: new Set(),
  configOverrides: new Map(),
  lastFramePoll: 0,
};

const RISK_CLS = { SAFE: "rl-safe", WARNING: "rl-warning", DANGER: "rl-danger", CRITICAL: "rl-critical" };
const RISK_W = { SAFE: 0, WARNING: 1, DANGER: 2, CRITICAL: 3 };
const PALETTE = ["#38BDF8", "#4ADE80", "#FBBF24", "#FB923C", "#A78BFA", "#F472B6", "#94A3B8"];

/* ============================================================
   THEME
   ============================================================ */
const Theme = {
  key: "autosense.theme",
  get() { return localStorage.getItem(this.key) || "dark"; },
  apply(mode, persist = true) {
    if (persist) localStorage.setItem(this.key, mode);
    let resolved = mode;
    if (mode === "system") resolved = matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    document.documentElement.dataset.theme = resolved;
    const dark = resolved === "dark";
    $("#icon-moon").style.display = dark ? "" : "none";
    $("#icon-sun").style.display = dark ? "none" : "";
    $$(`#theme-radios input`).forEach(r => r.checked = (r.value === mode));
  },
  cycle() { this.apply(document.documentElement.dataset.theme === "dark" ? "light" : "dark"); },
  init() {
    this.apply(this.get(), false);
    matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
      if (this.get() === "system") this.apply("system", false);
    });
    $("#btn-theme").addEventListener("click", () => this.cycle());
    $$("#theme-radios input").forEach(r => r.addEventListener("change", () => this.apply(r.value)));
  },
};

/* ============================================================
   CLOCK
   ============================================================ */
function initClock() {
  const ft = new Intl.DateTimeFormat("en-US", { hour: "2-digit", minute: "2-digit", hour12: true });
  const fd = new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "2-digit", month: "short", year: "numeric" });
  const tick = () => { const n = new Date(); $("#clock-time").textContent = ft.format(n); $("#clock-date").textContent = fd.format(n); };
  tick(); setInterval(tick, 1000);
}

/* ============================================================
   ROUTER
   ============================================================ */
const PAGES = ["dashboard", "video", "detection", "tracking", "distance", "safety", "viz", "model", "settings"];
function route(page) {
  if (!PAGES.includes(page)) page = "dashboard";
  $$(".nav-item").forEach(a => a.classList.toggle("active", a.dataset.page === page));
  PAGES.forEach(p => $(`#page-${p}`)?.classList.toggle("hidden", p !== page));
  if (location.hash !== `#${page}`) history.replaceState(null, "", `#${page}`);
  if (page === "model") loadModels();
  if (page === "video") loadSources();
  if (typeof setVizSource === "function") setVizSource();  // lazy viz preview
}
function initRouter() {
  $$(".nav-item").forEach(a => a.addEventListener("click", e => { e.preventDefault(); route(a.dataset.page); }));
  window.addEventListener("hashchange", () => route(location.hash.slice(1)));
  route(location.hash.slice(1) || "dashboard");
  $("#btn-settings").addEventListener("click", () => route("settings"));
  $("#btn-camera").addEventListener("click", () => route("video"));
}

/* ============================================================
   TELEMETRY
   ============================================================ */
function onStats(s) {
  State.stats = s;
  State.lastUpdate = Date.now();
  render(s);
}

function startWS() {
  let retries = 0;
  const connect = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    State.ws = ws;
    const ping = setInterval(() => { if (ws.readyState === 1) ws.send("ping"); }, 15000);
    ws.onmessage = ev => {
      try { const s = JSON.parse(ev.data); onStats(s); markTransport("WebSocket", "green"); }
      catch (e) {}
    };
    ws.onopen = () => { retries = 0; };
    ws.onerror = () => {};
    ws.onclose = () => {
      clearInterval(ping);
      startPolling();
      retries++;
      if (retries < 4) setTimeout(connect, 2000 * retries);  // retry quietly
    };
  };
  connect();
}

function startPolling() {
  if (State.pollTimer) return;
  markTransport("HTTP polling", "amber");
  State.pollTimer = setInterval(async () => {
    try { const s = await api("/api/status"); onStats(s); markTransport("HTTP polling", "amber"); }
    catch (e) { markTransport("Disconnected", "red"); $("#status-text").textContent = "Offline"; }
  }, 350);
}

function markTransport(t, dot) {
  State.transport = t;
  $("#conn-text").textContent = t === "WebSocket" ? "Live telemetry" : t === "HTTP polling" ? "Polling telemetry" : t;
  $("#conn-dot").className = `dot ${dot}`;
  $("#conn-transport").textContent = t;
}

/* ============================================================
   RENDER
   ============================================================ */
let chartTick = 0;

function render(s) {
  renderStatusPill(s);
  renderVideoHeader(s);
  renderOverlaysData(s);
  renderRightPanel(s);
  renderRecent(s);
  renderVideoPage(s);
  renderTables(s);
  renderSafety(s);
  renderModelCard(s);
  syncControls(s.config);

  chartTick++;
  if (chartTick % 2 === 0) {           // throttle canvas work to ~half rate
    drawDonut(s);
    drawSparkline(s);
    drawRiskStrip(s);
    drawTrajectory(s);
    drawRoiSchematic(s);
  }
}

function renderStatusPill(s) {
  const pill = $("#status-pill");
  const map = { live: "● Live Mode", paused: "● Paused", idle: "● Idle", error: "● System Error" };
  pill.dataset.state = s.mode;
  $("#status-text").textContent = (map[s.mode] || "● Idle").replace("● ", "");
  $("#conn-state").textContent = s.detail ? `${s.state} — ${s.detail}` : s.state;
  $("#conn-age").textContent = "0s";
}

function renderVideoHeader(s) {
  $("#vh-source").textContent = s.source.name || "No source";
  const streamFps = (typeof Video !== "undefined" && Video.fps) ? Video.fps() : 0;
  $("#vh-fps").textContent = `FPS: ${fmt(s.perf.fps, 1)}` +
    (streamFps > 0 ? ` · stream ${fmt(streamFps, 1)}` : "");
  $("#vh-res").textContent = s.source.width ? `${s.source.width} × ${s.source.height}` : "— × —";

  const roiBadge = $("#roi-frame-state");
  const roi = s.roi || { state: "unavailable", confidence: 0, uncertainty: true };
  const roiNames = {
    detected: "ROI DETECTED",
    low_confidence: "ROI LOW CONFIDENCE",
    reused: `ROI REUSED ${roi.frames_since_detection || ""}`.trim(),
    fallback: "ROI UNCERTAIN · FALLBACK",
    disabled: "ROI DISABLED · FALLBACK",
    unavailable: "ROI UNAVAILABLE",
  };
  roiBadge.dataset.state = roi.state || "fallback";
  roiBadge.title = roi.reason || "Classical computer-vision corridor state";
  roiBadge.querySelector("span:last-child").textContent =
    `${roiNames[roi.state] || "ROI UNCERTAIN"}${["detected", "low_confidence", "reused"].includes(roi.state) ? ` · ${fmt(roi.confidence, 2)}` : ""}`;
  roiBadge.classList.toggle("hidden", !(s.config?.overlays?.roi || s.config?.overlays?.lane));
  roiBadge.querySelector(".dot").className = `dot ${roi.uncertainty ? "amber" : "green"}`;

  const paused = s.state === "paused";
  $("#ico-pause").style.display = paused ? "none" : "";
  $("#ico-play").style.display = paused ? "" : "none";
  $("#paused-veil").classList.toggle("hidden", !paused);

  // risk banner
  const risk = s.safety.overall_risk;
  const show = RISK_W[risk] >= 1 && s.state === "running";
  const banner = $("#risk-banner");
  banner.classList.toggle("hidden", !show);
  if (show) {
    banner.dataset.risk = risk;
    $("#risk-banner-text").textContent =
      risk === "CRITICAL" ? `CRITICAL — AUTOMATIC BRAKE ADVISED` :
      risk === "DANGER" ? "DANGER — BRAKE ADVISED" : "PROXIMITY WARNING";
  }
}

function renderOverlaysData(s) {
  $("#ego-speed").textContent = s.ego_speed == null ? "—" : fmt0(s.ego_speed);
}

function renderRightPanel(s) {
  const c = s.counts;
  $("#st-vehicles").textContent = c.vehicles;
  $("#st-pedestrians").textContent = c.pedestrians;
  $("#st-twos").textContent = c.two_wheelers;
  $("#st-signs").textContent = c.traffic_signs;
  $("#st-other").textContent = c.other > 0 ? `+ ${c.other} other object(s)` : "";

  // modules
  const ml = $("#module-list");
  ml.innerHTML = Object.values(s.modules).map(m => `
    <div class="module-row">
      <span class="dot ${{ ok: "green", warn: "amber", err: "red", off: "gray" }[m.status] || "gray"}"></span>
      <div><div class="m-label">${m.label}</div><div class="m-value mono">${m.value}</div></div>
      <div class="m-sub ${m.status}" title="${m.title || ""}">${m.sub}</div>
    </div>`).join("");

  // performance
  $("#perf-fps").textContent = fmt(s.perf.fps, 1);
  $("#perf-inf").textContent = s.perf.inference_ms == null ? "N/A" : `${fmt0(s.perf.inference_ms)} ms`;
  $("#perf-latency").textContent = s.perf.processing_ms == null ? "N/A" : `${fmt0(s.perf.processing_ms)} ms`;
  $("#perf-input").textContent = `${s.perf.input_w} × ${s.perf.input_h}`;
  $("#perf-extra").textContent =
    `ROI ${s.perf.roi_ms == null ? "—" : `${fmt0(s.perf.roi_ms)} ms`} · res ${s.source.width}×${s.source.height} · source ${fmt(s.source.fps,1)} fps · ${s.perf.frames_processed} frames · up ${fmtUptime(s.perf.uptime_s)}`;

  // safety
  const box = $("#rp-risk");
  box.dataset.risk = s.safety.overall_risk;
  $("#rp-risk-text").textContent = s.safety.overall_risk;
  $("#rp-decision").textContent = s.safety.decision;
  $("#rp-closest").textContent = s.safety.closest_distance == null ? "—" : `${fmt(s.safety.closest_distance)} m`;
  $("#rp-ttc").textContent = s.safety.min_ttc == null ? "—" : `${fmt(s.safety.min_ttc)} s`;
}

function fmtUptime(sec) {
  sec = sec | 0;
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${sec / 60 | 0}m ${sec % 60}s`;
  return `${sec / 3600 | 0}h ${(sec % 3600) / 60 | 0}m`;
}

function renderRecent(s) {
  const list = $("#recent-list");
  $("#recent-count").textContent = s.recent.length ? `${s.recent.length} tracked` : "";
  if (!s.recent.length) {
    if (!list.dataset.empty) { list.innerHTML = `<div class="empty-note">${s.state === "running" ? "No objects tracked yet" : "Waiting for detections…"}</div>`; list.dataset.empty = "1"; }
    return;
  }
  list.dataset.empty = "";
  list.innerHTML = s.recent.map(r => `
    <div class="recent-item">
      ${r.thumb ? `<img src="${r.thumb}" alt="">` : `<div style="width:52px;height:52px;border-radius:8px;background:var(--accent-dim);display:grid;place-items:center;color:var(--accent);font-weight:700">${r.class_name[0].toUpperCase()}</div>`}
      <div>
        <div class="ri-cls">${r.class_name}</div>
        <div class="ri-sub mono">Confidence: ${fmt(r.confidence, 2)} · ${ago(r.ts)}</div>
      </div>
      <div class="ri-right">
        <div class="ri-dist mono">${r.distance == null ? "—" : fmt(r.distance) + " m"}</div>
        <div class="ri-id mono"><span class="risk-dot ${{SAFE:"rl-safe",WARNING:"rl-warning",DANGER:"rl-danger",CRITICAL:"rl-critical"}[r.risk] || "rl-safe"}" style="display:inline-block;width:7px;height:7px;border-radius:50%"></span>${r.track_id != null ? "ID " + r.track_id : ""}</div>
      </div>
    </div>`).join("");
}

function renderVideoPage(s) {
  $("#src-name").textContent = s.source.name || "No source selected";
  $("#src-meta").textContent = s.source.type
    ? `${s.source.type} · ${s.source.width}×${s.source.height} · ${fmt(s.source.fps, 1)} fps · ${s.state}`
    : "Select a sample, upload a video, or start the camera";
  if (s.source.type === "video" && s.source.duration_s > 0) {
    $("#src-progress").style.width = `${clamp(100 * s.source.position_s / s.source.duration_s, 0, 100)}%`;
    $("#src-time").textContent = `${fmt(s.source.position_s)}s / ${fmt(s.source.duration_s)}s`;
  } else {
    $("#src-progress").style.width = s.state === "running" ? "100%" : "0%";
    $("#src-time").textContent = s.source.type === "camera" ? "live stream" : "—";
  }
}

function riskPill(r) { return `<span class="risk-pill ${r}">${r}</span>`; }
function roiPill(d) {
  const relevance = d.roi_relevance || "uncertain";
  const labels = {
    in_corridor: "IN",
    near_corridor: "NEAR",
    outside_corridor: "OUTSIDE",
    uncertain: "UNCERTAIN",
  };
  const title = d.roi_label || "ROI relevance unavailable";
  return `<span class="roi-pill ${relevance}" title="${title}">${labels[relevance] || "UNCERTAIN"}</span>`;
}

function renderTables(s) {
  // detections
  const det = s.detections;
  $("#det-total").textContent = `${s.detection_total} in frame`;
  $("#det-empty").style.display = det.length ? "none" : "";
  $("#det-table tbody").innerHTML = det.map(d => `
    <tr>
      <td style="text-transform:capitalize">${d.class_name}</td>
      <td class="mono">${fmt(d.confidence, 2)}</td>
      <td class="mono">${d.track_id ?? "—"}</td>
      <td class="mono">${d.distance == null ? "—" : fmt(d.distance) + " m"}</td>
      <td class="mono">${d.ttc == null ? "—" : fmt(d.ttc) + " s"}</td>
      <td>${riskPill(d.risk)}</td>
      <td>${roiPill(d)}</td>
    </tr>`).join("");

  // tracks
  const tr = s.tracks;
  $("#trk-active").textContent = tr.length;
  $("#trk-count2").textContent = `${tr.length} active`;
  $("#trk-mode").textContent = s.config.tracking ? (s.config.tracker.includes("bytetrack") ? "ByteTrack" : "BoT-SORT") : "Off";
  $("#trk-empty").style.display = tr.length ? "none" : "";
  $("#trk-table tbody").innerHTML = tr.map(t => `
    <tr><td class="mono">#${t.id}</td><td style="text-transform:capitalize">${t.class_name}</td>
      <td class="mono">${t.distance == null ? "—" : fmt(t.distance) + " m"}</td>
      <td class="mono">${fmt(t.age_s)} s</td><td class="mono">${t.seen_frames}</td><td>${roiPill(t)}</td><td>${riskPill(t.risk)}</td></tr>`).join("");

  // distances
  $("#dist-empty").style.display = det.length ? "none" : "";
  $("#dist-table tbody").innerHTML = det.map(d => `
    <tr><td style="text-transform:capitalize">${d.class_name}</td>
      <td class="mono">${d.track_id ?? "—"}</td>
      <td class="mono">${d.bbox[3] - d.bbox[1]} px</td>
      <td class="mono"><b>${d.distance == null ? "—" : fmt(d.distance) + " m"}</b></td>
      <td>${roiPill(d)}</td></tr>`).join("");

  // Current geometry, confidence and explicit uncertainty state.
  const roi = s.roi || { state: "unavailable", confidence: 0, uncertainty: true, reason: "No frame" };
  const roiStatus = $("#roi-live-status");
  if (roiStatus) {
    const stateLabel = {
      detected: "DETECTED",
      low_confidence: "LOW CONFIDENCE",
      reused: `REUSED · ${roi.frames_since_detection || 0} frame(s)`,
      fallback: "UNCERTAIN · FALLBACK ROI",
      disabled: "DYNAMIC OFF · FALLBACK ROI",
      unavailable: "WAITING FOR FRAME",
    }[roi.state] || "UNCERTAIN";
    roiStatus.dataset.state = roi.state || "fallback";
    roiStatus.textContent = `${stateLabel} · confidence ${fmt(roi.confidence, 2)}`;
    roiStatus.title = roi.reason || "Classical-CV ROI status";
  }
  const roiPolygon = s.static_roi_polygon || s.roi_polygon;
  const corners = roiPolygon?.corners?.length ? roiPolygon.corners : roiPolygon?.points?.slice(0, 4);
  if (corners?.length) {
    $("#roi-coords").textContent = corners.map((p, i) =>
      `${["TL", "TR", "BR", "BL"][i]} (${p[0]}, ${p[1]})`).join(" · ");
  }
}

function renderSafety(s) {
  const hero = $("#safety-hero");
  hero.dataset.risk = s.safety.overall_risk;
  $("#sh-risk").textContent = s.safety.overall_risk;
  $("#sh-decision").textContent = s.safety.decision;
  $("#sh-closest").textContent = s.safety.closest_distance == null ? "—" : `${fmt(s.safety.closest_distance)} m`;
  $("#sh-ttc").textContent = s.safety.min_ttc == null ? "—" : `${fmt(s.safety.min_ttc)} s`;

  const th = s.safety.thresholds;
  $("#threshold-table tbody").innerHTML = `
    <tr><td><span class="risk-pill WARNING">WARNING</span></td><td class="mono">${th.distance.warning} m</td><td class="mono">${th.ttc.warning} s</td></tr>
    <tr><td><span class="risk-pill DANGER">DANGER</span></td><td class="mono">${th.distance.danger} m</td><td class="mono">${th.ttc.danger} s</td></tr>
    <tr><td><span class="risk-pill CRITICAL">CRITICAL</span></td><td class="mono">${th.distance.critical} m</td><td class="mono">${th.ttc.critical} s</td></tr>`;

  const rows = s.detections.filter(d => d.ttc != null || d.in_path);
  $("#ttc-empty").style.display = rows.length ? "none" : "";
  $("#ttc-table tbody").innerHTML = rows
    .sort((a, b) => RISK_W[b.risk] - RISK_W[a.risk])
    .map(d => `
      <tr><td class="mono">${d.track_id ?? "—"}</td>
        <td style="text-transform:capitalize">${d.class_name}</td>
        <td class="mono">${d.distance == null ? "—" : fmt(d.distance) + " m"}</td>
        <td class="mono">${fmt(d.closing_speed, 2)} m/s</td>
        <td class="mono"><b>${d.ttc == null ? "—" : fmt(d.ttc) + " s"}</b></td>
        <td>${riskPill(d.risk)}</td></tr>`).join("");
}

function renderModelCard(s) {
  $("#model-name").textContent = s.model.name;
  $("#model-device").textContent = s.model.device;

  /* header device badge mirrors the actual compute device */
  const db = $("#btn-device");
  if (db) {
    const gpu = /gpu/i.test(s.model.device || "");
    if (db._gpu !== gpu) {
      db._gpu = gpu;
      db.classList.toggle("gpu", gpu);
      db.textContent = gpu ? "GPU" : "CPU";
    }
    db.title = `${s.model.device} · setting: ${s.config.device || "auto"} — click to switch to ${gpu ? "CPU" : "GPU"}`;
  }
  $("#model-input").textContent = `${s.config.imgsz} × ${s.config.imgsz}`;
  $("#model-infer").textContent = s.perf.inference_ms == null ? "—" : `${fmt0(s.perf.inference_ms)} ms`;
  $("#model-class-count").textContent = s.model.classes.length;
  $("#model-classes").innerHTML = s.model.classes.map(c => `<span class="chip mono">${c}</span>`).join("") || `<span class="muted">—</span>`;
}

/* ============================================================
   CONTROL SYNC (stats -> UI, without fighting the user)
   ============================================================ */
function bindEditing(el) {
  el.addEventListener("pointerdown", () => State.editing.add(el.id));
  el.addEventListener("focus", () => State.editing.add(el.id));
  el.addEventListener("change", () => setTimeout(() => State.editing.delete(el.id), 400));
  el.addEventListener("blur", () => State.editing.delete(el.id));
}

function syncControls(cfg) {
  if (!cfg) return;
  const S = State.editing;
  const effective = JSON.parse(JSON.stringify(cfg));
  for (const [path, override] of State.configOverrides) {
    const echoed = getConfigPath(cfg, path);
    if (sameConfigValue(echoed, override.value)) {
      State.configOverrides.delete(path);
    } else {
      setConfigPath(effective, path, override.value);
    }
  }
  cfg = effective;

  if (!S.has("conf-slider")) { $("#conf-slider").value = Math.round(cfg.conf * 100); $("#conf-val").textContent = fmt(cfg.conf, 2); }
  if (!S.has("iou-slider"))  { $("#iou-slider").value = Math.round(cfg.iou * 100);  $("#iou-val").textContent = fmt(cfg.iou, 2); }
  if (!S.has("focal-slider")) { $("#focal-slider").value = cfg.focal_length; $("#focal-val").textContent = cfg.focal_length; }
  if (!S.has("smooth-slider")) { $("#smooth-slider").value = cfg.smoothing; $("#smooth-val").textContent = cfg.smoothing; }

  if (!S.has("roi-ego"))    { $("#roi-ego").value = Math.round(cfg.roi.ego_center_ratio * 100);   $("#roi-ego-val").textContent = fmt(cfg.roi.ego_center_ratio, 2); }
  if (!S.has("roi-top"))    { $("#roi-top").value = Math.round(cfg.roi.top_width_ratio * 100);    $("#roi-top-val").textContent = fmt(cfg.roi.top_width_ratio, 2); }
  if (!S.has("roi-bottom")) { $("#roi-bottom").value = Math.round(cfg.roi.bottom_width_ratio * 100); $("#roi-bottom-val").textContent = fmt(cfg.roi.bottom_width_ratio, 2); }
  if (!S.has("roi-hs"))     { $("#roi-hs").value = Math.round(cfg.roi.height_start * 100);        $("#roi-hs-val").textContent = fmt(cfg.roi.height_start, 2); }
  if (!S.has("roi-dynamic")) $("#roi-dynamic").checked = !!cfg.roi.dynamic_enabled;

  const roiRange = (id, key, percent = false, digits = 2) => {
    if (S.has(id)) return;
    const raw = cfg.roi[key];
    if (raw == null) return;
    const value = percent ? Math.round(raw * 100) : Math.round(raw);
    $(`#${id}`).value = value;
    $(`#${id}-val`).textContent = percent ? fmt(raw, digits) : String(value);
  };
  roiRange("roi-search-center", "search_center_ratio", true);
  roiRange("roi-search-top", "search_top_ratio", true);
  roiRange("roi-search-top-width", "search_top_width_ratio", true);
  roiRange("roi-search-bottom-width", "search_bottom_width_ratio", true);
  roiRange("roi-canny-low", "canny_low", false);
  roiRange("roi-canny-high", "canny_high", false);
  roiRange("roi-hough", "hough_threshold", false);
  roiRange("roi-min-conf", "min_confidence", true);
  roiRange("roi-smoothing", "smoothing", true);
  roiRange("roi-reuse", "max_reuse_frames", false);

  $$("#imgsz-seg button").forEach(b => b.classList.toggle("on", +b.dataset.v === cfg.imgsz));
  $("#imgsz-val").textContent = cfg.imgsz;
  $$("#perf-seg button").forEach(b => b.classList.toggle("on", +b.dataset.v === cfg.imgsz));
  $$("#device-seg button").forEach(b => b.classList.toggle("on", b.dataset.v === (cfg.device || "auto")));

  /* header Turbo button mirrors the live input size */
  const turbo = cfg.imgsz <= 320;
  if (syncControls._turbo !== turbo) {
    syncControls._turbo = turbo;
    const tb = $("#btn-turbo");
    if (tb) tb.classList.toggle("on", turbo);
    localStorage.setItem("autosense.turbo", turbo ? "1" : "0");
  }
  $$("#tracker-seg button").forEach(b => b.classList.toggle("on", b.dataset.v === cfg.tracker));
  $("#trk-enable").checked = cfg.tracking;
  $("#trk-ids").checked = cfg.overlays.ids;

  // overlay checkboxes (menu + viz page)
  $$(`[data-ov]`).forEach(cb => { cb.checked = !!cfg.overlays[cb.dataset.ov]; });
}

/* Turbo FPS = inference at 320px input (max FPS) vs 640px (max accuracy).
   Single helper shared by the header toggle and the boot re-apply; the
   choice is persisted client-side because the backend keeps imgsz in
   memory only (it resets to 640 whenever the server restarts). */
function applyTurbo(on, announce = true) {
  localStorage.setItem("autosense.turbo", on ? "1" : "0");
  api("/api/config", "POST", { imgsz: on ? 320 : 640 })
    .then(() => { if (announce) toast(on ? "Turbo FPS on — 320px input, maximum frame rate" : "Turbo FPS off — 640px input, maximum accuracy", "ok"); })
    .catch(e => { if (announce) toast(e.message, "err"); });
}

function initControls() {
  ["conf-slider", "iou-slider", "focal-slider", "smooth-slider", "roi-ego", "roi-top", "roi-bottom", "roi-hs",
   "roi-search-center", "roi-search-top", "roi-search-top-width", "roi-search-bottom-width", "roi-canny-low", "roi-canny-high",
   "roi-hough", "roi-min-conf", "roi-smoothing", "roi-reuse"].forEach(id => bindEditing($("#" + id)));

  $("#conf-slider").addEventListener("input", e => { $("#conf-val").textContent = (e.target.value / 100).toFixed(2); postCfg({ conf: e.target.value / 100 }); });
  $("#iou-slider").addEventListener("input", e => { $("#iou-val").textContent = (e.target.value / 100).toFixed(2); postCfg({ iou: e.target.value / 100 }); });
  $("#focal-slider").addEventListener("input", e => { $("#focal-val").textContent = e.target.value; postCfg({ focal_length: +e.target.value }); });
  $("#smooth-slider").addEventListener("input", e => { $("#smooth-val").textContent = e.target.value; postCfg({ smoothing: +e.target.value }); });
  $("#roi-ego").addEventListener("input", e => { $("#roi-ego-val").textContent = (e.target.value / 100).toFixed(2); postCfg({ roi: { ego_center_ratio: e.target.value / 100 } }); });
  $("#roi-top").addEventListener("input", e => { $("#roi-top-val").textContent = (e.target.value / 100).toFixed(2); postCfg({ roi: { top_width_ratio: e.target.value / 100 } }); });
  $("#roi-bottom").addEventListener("input", e => { $("#roi-bottom-val").textContent = (e.target.value / 100).toFixed(2); postCfg({ roi: { bottom_width_ratio: e.target.value / 100 } }); });
  $("#roi-hs").addEventListener("input", e => { $("#roi-hs-val").textContent = (e.target.value / 100).toFixed(2); postCfg({ roi: { height_start: e.target.value / 100 } }); });

  $("#roi-dynamic").addEventListener("change", e => {
    postCfg({ roi: { dynamic_enabled: e.target.checked } });
    toast(e.target.checked ? "Classical dynamic ROI enabled" : "Using configured fallback ROI", "ok");
  });
  const roiPercent = (id, key) => $("#" + id).addEventListener("input", e => {
    const value = Number(e.target.value) / 100;
    $("#" + id + "-val").textContent = value.toFixed(2);
    postCfg({ roi: { [key]: value } });
  });
  const roiInteger = (id, key) => $("#" + id).addEventListener("input", e => {
    const value = Number(e.target.value);
    $("#" + id + "-val").textContent = String(value);
    postCfg({ roi: { [key]: value } });
  });
  roiPercent("roi-search-center", "search_center_ratio");
  roiPercent("roi-search-top", "search_top_ratio");
  roiPercent("roi-search-top-width", "search_top_width_ratio");
  roiPercent("roi-search-bottom-width", "search_bottom_width_ratio");
  roiInteger("roi-canny-low", "canny_low");
  roiInteger("roi-canny-high", "canny_high");
  roiInteger("roi-hough", "hough_threshold");
  roiPercent("roi-min-conf", "min_confidence");
  roiPercent("roi-smoothing", "smoothing");
  roiInteger("roi-reuse", "max_reuse_frames");

  $$("#imgsz-seg button").forEach(b => b.addEventListener("click", () => {
    localStorage.setItem("autosense.turbo", +b.dataset.v <= 320 ? "1" : "0");
    api("/api/config", "POST", { imgsz: +b.dataset.v });
  }));

  // performance mode preset (input size 320 = max FPS on CPU, 640 = max accuracy)
  $$("#perf-seg button").forEach(b => b.addEventListener("click", () => {
    localStorage.setItem("autosense.turbo", +b.dataset.v <= 320 ? "1" : "0");
    api("/api/config", "POST", { imgsz: +b.dataset.v })
      .then(() => toast(`Input size ${b.dataset.v} — ${+b.dataset.v <= 416 ? "higher FPS" : "higher accuracy"}`, "ok"))
      .catch(e => toast(e.message, "err"));
  }));

  // Turbo FPS toggle in the video header (persisted preference, synced with presets)
  $("#btn-turbo").addEventListener("click", () => applyTurbo(!$("#btn-turbo").classList.contains("on")));

  // compute device selection (auto / gpu / cpu) — the engine reloads the model
  // onto the requested device; the header badge shows what is actually running
  $$("#device-seg button").forEach(b => b.addEventListener("click", () => {
    api("/api/config", "POST", { device: b.dataset.v })
      .then(() => toast(`Compute device: ${b.dataset.v === "auto" ? "Auto" : b.dataset.v.toUpperCase()} — reloading model`, "ok"))
      .catch(e => toast(e.message, "err"));
  }));

  // header badge doubles as a quick GPU <-> CPU toggle
  $("#btn-device").addEventListener("click", () => {
    const toGpu = !$("#btn-device").classList.contains("gpu");
    api("/api/config", "POST", { device: toGpu ? "gpu" : "cpu" })
      .then(() => toast(`Switching compute device to ${toGpu ? "GPU" : "CPU"}…`, "ok"))
      .catch(e => toast(e.message, "err"));
  });

  // camera device probing
  $("#cam-detect").addEventListener("click", () => detectCameras());
  $$("#tracker-seg button").forEach(b => b.addEventListener("click", () => api("/api/config", "POST", { tracker: b.dataset.v, tracking: true })));
  $("#trk-enable").addEventListener("change", e => api("/api/config", "POST", { tracking: e.target.checked }));
  $("#trk-ids").addEventListener("change", e => api("/api/config", "POST", { overlays: { ids: e.target.checked } }));

  $$(`[data-ov]`).forEach(cb => cb.addEventListener("change", () => {
    if (cb.disabled) return;
    postCfg({ overlays: { [cb.dataset.ov]: cb.checked } });
  }));

  // video transport buttons
  $("#btn-pause").addEventListener("click", () => {
    const st = State.stats?.state;
    api(`/api/control/${st === "paused" ? "resume" : "pause"}`, "POST").catch(e => toast(e.message, "err"));
  });
  $("#btn-restart").addEventListener("click", () => api("/api/control/restart", "POST"));
  $("#ctrl-play").addEventListener("click", () => api("/api/control/start", "POST"));
  $("#ctrl-pause").addEventListener("click", () => api("/api/control/pause", "POST"));
  $("#ctrl-resume").addEventListener("click", () => api("/api/control/resume", "POST"));
  $("#ctrl-restart").addEventListener("click", () => api("/api/control/restart", "POST"));
  $("#ctrl-stop").addEventListener("click", () => api("/api/control/stop", "POST"));

  // overlay menu popover
  const menu = $("#overlay-menu");
  $("#btn-overlay-menu").addEventListener("click", e => { e.stopPropagation(); menu.classList.toggle("hidden"); });
  document.addEventListener("click", e => { if (!menu.contains(e.target)) menu.classList.add("hidden"); });

  // fullscreen
  $("#btn-fullscreen").addEventListener("click", () => {
    const stage = $("#video-stage");
    if (document.fullscreenElement) document.exitFullscreen();
    else stage.requestFullscreen?.();
  });
  document.addEventListener("fullscreenchange", () => $("#video-stage").classList.toggle("fullscreen", !!document.fullscreenElement));

  // camera
  $("#cam-start").addEventListener("click", () => {
    api("/api/source/camera", "POST", { index: +$("#cam-index").value || 0 })
      .then(() => { toast("Camera source requested", "ok"); route("dashboard"); })
      .catch(e => toast(e.message, "err"));
  });
  $("#cam-stop").addEventListener("click", () => api("/api/control/stop", "POST"));

  // quick actions
  $("#qa-distance").addEventListener("click", () => route("distance"));
  $("#qa-risk").addEventListener("click", () => route("safety"));
  $("#qa-tracks").addEventListener("click", () => route("tracking"));
  $("#qa-report").addEventListener("click", downloadReport);

  // settings
  $("#btn-reset-settings").addEventListener("click", () => {
    localStorage.removeItem(Theme.key); localStorage.removeItem("autosense.stream");
    localStorage.removeItem("autosense.turbo");
    Theme.apply("dark"); setStreamMode("mjpeg", false);
    $$("#stream-seg button").forEach(b => b.classList.toggle("on", b.dataset.v === "mjpeg"));
    toast("UI preferences reset", "ok");
  });
  $$("#stream-seg button").forEach(b => b.addEventListener("click", () => {
    setStreamMode(b.dataset.v);
    $$("#stream-seg button").forEach(x => x.classList.toggle("on", x === b));
  }));
}

/* ============================================================
   VIDEO STREAM TRANSPORT
   The perception overlays (boxes / IDs / labels / HUD) are baked
   into the server's annotated JPEG frames, so the frontend must
   display every frame at full brightness — never a stalled,
   half-decoded, whitened <img> surface.

   Rendering hierarchy (per spec):
     video layer  <img id="video-img">  (fallback surface)
       ↓
     transparent visualization canvas <canvas id="video-canvas">
       (cleared with ctx.clearRect — never filled white)
       ↓
     optional transparent HUD elements (layers, speed, banner)

   The canvas has NO opaque background and is never filled white:
   every frame is cleared with clearRect() and re-painted, so a
   stalled stream simply shows the last good frame transparently
   over the video layer instead of a white sheet.
   ============================================================ */
const Video = {
  mode: "mjpeg",            // active transport: mjpeg | polling | img
  controller: null,         // AbortController of the active fetch stream
  pollTimer: null,
  stallTimer: null,
  retryTimer: null,
  lastFrameTs: 0,
  retries: 0,
  failCount: 0,
  running: false,
  canvas: null,
  ctx: null,
  img: null,
  times: [],

  init() {
    this.canvas = $("#video-canvas");
    this.img = $("#video-img");
    this.ctx = this.canvas.getContext("2d", { alpha: true });
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  },

  /* delivered stream FPS (frames painted over the last 4s window) */
  fps() {
    const now = Date.now();
    while (this.times.length && now - this.times[0] > 4000) this.times.shift();
    if (this.times.length < 2) return 0;
    const elapsed = Math.min(4, (now - this.times[0]) / 1000);
    return elapsed > 0.05 ? (this.times.length - 1) / elapsed : 0;
  },

  /* ---------- painting (transparent, never white) ---------- */
  paint(source, w, h) {
    const cv = this.canvas;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    /* transparent clear — no fillRect, no white background, ever */
    this.ctx.clearRect(0, 0, cv.width, cv.height);
    this.ctx.drawImage(source, 0, 0, w, h);
    this.lastFrameTs = Date.now();
    this.times.push(this.lastFrameTs);
    if (this.times.length > 240) this.times.splice(0, this.times.length - 240);
    /* first good frame: canvas takes over as the display surface */
    this.img.classList.add("stream-ok");
    this.failCount = 0;
  },

  clearSurface() {
    if (this.ctx) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  },
  /* ---------- transport selection ---------- */
  async setMode(mode, persist = true) {
    if (persist) localStorage.setItem("autosense.stream", mode);
    this.stop();
    if (mode === "mjpeg") {
      await this.startDemuxer();
    } else {
      this.startPolling();
    }
    $$("#stream-seg button").forEach(b =>
      b.classList.toggle("on", b.dataset.v === (mode === "img" ? "mjpeg" : mode)));
  },

  stop() {
    this.running = false;
    if (this.controller) { try { this.controller.abort(); } catch (e) {} this.controller = null; }
    clearInterval(this.pollTimer); this.pollTimer = null;
    clearInterval(this.stallTimer); this.stallTimer = null;
    clearTimeout(this.retryTimer); this.retryTimer = null;
  },

  /* ---------- primary: fetch-based MJPEG demuxer ---------- */
  async startDemuxer() {
    if (!window.fetch || !window.AbortController || !window.createImageBitmap) {
      this.startLegacyImg();   // very old browser — stream straight into <img>
      return;
    }
    this.mode = "mjpeg";
    this.running = true;
    this.retries = 0;
    this.armStallWatchdog();
    /* poster frame under the transparent canvas until real frames flow */
    this.img.src = `/api/video/frame?t=${Date.now()}`;
    setVizSource();

    while (this.running && this.retries < 4) {
      await this.pumpStream();
      if (!this.running) break;
      this.retries++;
      if (this.retries >= 4) { this.startLegacyImg(); return; }
      await new Promise(r => { this.retryTimer = setTimeout(r, 400 * this.retries); });
    }
  },
  /* ---------- MJPEG multipart pump ----------
     Server sends:  --frame\r\nContent-Type: image/jpeg\r\n\r\n<jpeg>\r\n
     We parse boundaries, decode and paint the newest complete frame. */
  async pumpStream() {
    this.controller = new AbortController();
    const BOUND = new TextEncoder().encode("--frame");
    const HDREND = new TextEncoder().encode("\r\n\r\n");
    let buf = new Uint8Array(0);
    try {
      const res = await fetch("/api/video/stream", { signal: this.controller.signal, cache: "no-store" });
      if (!res.ok || !res.body) throw new Error(`stream HTTP ${res.status}`);
      const reader = res.body.getReader();
      while (this.running) {
        const { done, value } = await reader.read();
        if (done) break;
        const merged = new Uint8Array(buf.length + value.length);
        merged.set(buf, 0); merged.set(value, buf.length);
        buf = merged;

        /* extract every complete frame buffered; paint only the newest
           (keeps latency at one frame even if the network bursts) */
        let newest = null;
        for (;;) {
          const b0 = findSeq(buf, BOUND);
          if (b0 < 0) { if (buf.length > (8 << 20)) buf = new Uint8Array(0); break; }
          const he = findSeq(buf, HDREND, b0);
          if (he < 0) { if (b0 > 0) buf = buf.slice(b0); break; }
          const b1 = findSeq(buf, BOUND, he + 4);
          if (b1 < 0) { if (b0 > 0) buf = buf.slice(b0); break; } // frame still arriving
          let end = b1;
          if (end >= 2 && buf[end - 1] === 10 && buf[end - 2] === 13) end -= 2; // trailing \r\n
          if (end - (he + 4) > 4) newest = buf.slice(he + 4, end);
          buf = buf.slice(b1); // keep the next boundary marker for the next pass
        }
        if (newest) await this.decodeAndPaint(newest);
      }
    } catch (e) { /* aborted or network hiccup — caller decides to retry */ }
    finally { this.controller = null; }
  },

  async decodeAndPaint(bytes) {
    try {
      const blob = new Blob([bytes], { type: "image/jpeg" });
      const bmp = await createImageBitmap(blob);
      this.paint(bmp, bmp.width, bmp.height);
      if (bmp.close) bmp.close();
    } catch (e) {
      this.failCount++;
      if (this.failCount > 30) this.startLegacyImg(); // decoder poisoned → plain <img>
    }
  },
  /* ---------- fallback: snapshot polling ----------
     The multipart-into-<img> path is the one that white-stalls in
     Chromium, so it is NEVER used — every fallback repaints through
     the transparent canvas instead. */
  startLegacyImg() {
    if (this.mode === "polling" && this.running) return; // already on the safe path
    this.stop();
    this.img.classList.remove("stream-ok");
    this.clearSurface();
    this.img.src = `/api/video/frame?t=${Date.now()}`; // single-frame poster only
    toast("Stream decoder unavailable — switched to snapshot polling");
    this.startPolling();
  },

  /* ---------- snapshot polling transport ---------- */
  startPolling() {
    this.running = true;
    this.mode = "polling";
    this.armStallWatchdog();
    let tickN = 0;
    const tick = async () => {
      try {
        const r = await fetch(`/api/video/frame?t=${Date.now()}`, { cache: "no-store" });
        if (!r.ok) return;
        const blob = await r.blob();
        if (window.createImageBitmap) {
          const bmp = await createImageBitmap(blob);
          this.paint(bmp, bmp.width, bmp.height);
          if (bmp.close) bmp.close();
        } else {
          const url = URL.createObjectURL(blob);
          const im = new Image();
          im.onload = () => { this.paint(im, im.naturalWidth, im.naturalHeight); URL.revokeObjectURL(url); };
          im.src = url;
        }
        const viz = $("#viz-img");
        if (viz && vizActive() && (tickN % 4 === 0)) viz.src = `/api/video/frame?t=${Date.now()}`;
      } catch (e) {}
      tickN++;
    };
    tick();
    this.pollTimer = setInterval(tick, 140);
  },

  /* ---------- watchdog: silently restart a stalled stream ---------- */
  armStallWatchdog(legacy = false) {
    clearInterval(this.stallTimer);
    this.lastFrameTs = Date.now();
    let busy = false;
    this.stallTimer = setInterval(async () => {
      if (!this.running || busy) return;
      if (Date.now() - this.lastFrameTs < 3000) return;
      busy = true;
      try {
        const s = await api("/api/status");
        if (s.state === "running" && Date.now() - this.lastFrameTs >= 3000) {
          if (legacy || this.mode === "img") {
            this.img.src = `/api/video/stream?t=${Date.now()}`;
            this.lastFrameTs = Date.now();
          } else {
            this.lastFrameTs = Date.now();
            this.setMode(this.mode === "mjpeg" ? "mjpeg" : "polling", false); // re-arms the watchdog
          }
        } else if (s.state !== "running") {
          this.lastFrameTs = Date.now(); // paused/idle — nothing should flow
        }
      } catch (e) { this.lastFrameTs = Date.now(); }
      busy = false;
    }, 1500);
  },
};

/* lazy Visualizations-page preview: only connects while that page is open */
function vizActive() {
  const p = document.querySelector("#page-viz");
  return !!p && !p.classList.contains("hidden");
}
function setVizSource() {
  const viz = $("#viz-img");
  if (!viz) return;
  if (vizActive()) {
    viz.src = Video.mode === "polling"
      ? `/api/video/frame?t=${Date.now()}`
      : `/api/video/stream?t=${Date.now() + 7}`;
  } else {
    viz.removeAttribute("src");
  }
}

let framePollTimer = null; // kept for backward compatibility
function setStreamMode(mode, persist = true) {
  Video.setMode(mode, persist);
}

/* ============================================================
   SOURCES & MODELS LISTS
   ============================================================ */
function updateCameraOptions(found) {
  const sel = $("#cam-index");
  if (!sel) return;
  sel.innerHTML = found.length
    ? found.map(i => `<option value="${i}">Camera ${i} — detected</option>`).join("")
    : [0, 1, 2, 3].map(i => `<option value="${i}">Camera ${i}</option>`).join("");
}

let camDetectBusy = false;
async function detectCameras() {
  if (camDetectBusy) return;
  camDetectBusy = true;
  const btn = $("#cam-detect"), note = $("#cam-note"), sel = $("#cam-index");
  const prevName = State.stats?.source?.name;
  const prevType = State.stats?.source?.type;
  const wasRunning = State.stats?.state === "running";
  const original = btn.textContent;
  btn.disabled = true; btn.textContent = "Detecting…";
  const found = [];
  try {
    for (let i = 0; i <= 5; i++) {
      btn.textContent = `Probing camera ${i}…`;
      try { await api("/api/source/camera", "POST", { index: i }); }
      catch (e) { break; } // server rejected — stop probing
      await new Promise(r => setTimeout(r, 900)); // give the capture thread time to open the device
      try {
        const s = await api("/api/status");
        if (s.source?.type === "camera" && s.state !== "error") found.push(i);
      } catch (e) {}
    }
  } finally {
    // restore the previous source so the dashboard keeps playing
    try {
      if (prevType === "video" && prevName) {
        await api("/api/source/sample", "POST", { name: prevName });
      } else if (wasRunning) {
        await api("/api/control/stop", "POST");
      }
    } catch (e) {}
  }
  updateCameraOptions(found);
  if (found.length) sel.value = String(found[0]);
  note.innerHTML = found.length
    ? `Detected <b>${found.length}</b> camera device(s) — index ${found.join(", ")}. Pick one and press <b>Start Camera</b>.`
    : "No cameras detected on indices 0–5. Connect a device to the server and retry.";
  toast(found.length ? `${found.length} camera(s) detected` : "No cameras detected", found.length ? "ok" : "");
  btn.disabled = false; btn.textContent = original;
  camDetectBusy = false;
}

async function loadSources() {
  try {
    const d = await api("/api/sources");
    const cur = State.stats?.source?.name;
    $("#sample-list").innerHTML = d.videos.map(v => `
      <div class="src-item ${v.name === cur ? "active" : ""}">
        <div><div class="si-name">${v.name}</div><div class="si-meta mono">${v.kind} · ${v.size_mb} MB</div></div>
        <button class="btn accent" data-load="${v.name}">Load</button>
      </div>`).join("") || `<div class="empty-note">No videos available</div>`;
    $$(`#sample-list [data-load]`).forEach(b => b.addEventListener("click", async () => {
      try { await api("/api/source/sample", "POST", { name: b.dataset.load }); toast(`Loaded ${b.dataset.load}`, "ok"); route("dashboard"); }
      catch (e) { toast(e.message, "err"); }
    }));
  } catch (e) {}
}

async function loadModels() {
  try {
    const d = await api("/api/models");
    const sel = $("#det-model");
    sel.innerHTML = d.models.map(m => `<option value="${m.name}" ${m.name === d.active ? "selected" : ""}>${m.name}</option>`).join("");
    sel.onchange = () => api("/api/models/select", "POST", { name: sel.value }).then(() => toast("Model switched — reloading weights", "ok"));
    $("#model-list").innerHTML = d.models.map(m => `
      <div class="src-item ${m.name === d.active ? "active" : ""}">
        <div><div class="si-name mono">${m.name}</div><div class="si-meta mono">${m.size_mb} MB</div></div>
        ${m.name === d.active ? `<span class="m-sub ok" style="font-size:11px;font-weight:700">ACTIVE</span>` : `<button class="btn accent" data-model="${m.name}">Select</button>`}
      </div>`).join("");
    $$(`#model-list [data-model]`).forEach(b => b.addEventListener("click", async () => {
      await api("/api/models/select", "POST", { name: b.dataset.model });
      toast("Model switched", "ok"); loadModels();
    }));
  } catch (e) {}
}

/* ============================================================
   UPLOADS
   ============================================================ */
function uploadFile(url, file, statusEl, done) {
  const fd = new FormData();
  fd.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", url);
  xhr.upload.onprogress = e => { if (e.lengthComputable) statusEl.textContent = `uploading… ${(100 * e.loaded / e.total).toFixed(0)}%`; };
  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) { statusEl.textContent = "done"; done && done(JSON.parse(xhr.responseText)); }
    else { let m = "upload failed"; try { m = JSON.parse(xhr.responseText).error || m; } catch {} statusEl.textContent = m; toast(m, "err"); }
    setTimeout(() => statusEl.textContent = "", 6000);
  };
  xhr.onerror = () => { statusEl.textContent = "network error"; };
  xhr.send(fd);
  statusEl.textContent = "uploading… 0%";
}

function initUploads() {
  const dz = $("#video-drop"), fi = $("#video-file"), st = $("#video-upload-status");
  fi.addEventListener("change", () => fi.files[0] && uploadFile("/api/source/video", fi.files[0], st, () => { toast("Video loaded", "ok"); route("dashboard"); }));
  ["dragover", "dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => {
    e.preventDefault(); dz.classList.toggle("drag", ev === "dragover");
    if (ev === "drop" && e.dataTransfer.files[0]) uploadFile("/api/source/video", e.dataTransfer.files[0], st, () => { toast("Video loaded", "ok"); route("dashboard"); });
  }));

  const mdz = $("#model-drop"), mfi = $("#model-file"), mst = $("#model-upload-status");
  const selectUploadedModel = result => {
    if (!result?.name) { loadModels(); return; }
    api("/api/models/select", "POST", { name: result.name })
      .then(() => toast(`Uploaded checkpoint selected: ${result.name}`, "ok"))
      .catch(e => toast(`Weights uploaded, but not selected: ${e.message}`, "err"))
      .finally(loadModels);
  };
  mfi.addEventListener("change", () => mfi.files[0] && uploadFile("/api/models/upload", mfi.files[0], mst, selectUploadedModel));
  ["dragover", "dragleave", "drop"].forEach(ev => mdz.addEventListener(ev, e => {
    e.preventDefault(); mdz.classList.toggle("drag", ev === "dragover");
    if (ev === "drop" && e.dataTransfer.files[0]) uploadFile("/api/models/upload", e.dataTransfer.files[0], mst, selectUploadedModel);
  }));
}

async function downloadReport() {
  try {
    const r = await api("/api/report");
    const blob = new Blob([JSON.stringify(r, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `autosense_report_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
    a.click(); URL.revokeObjectURL(a.href);
    toast("Report generated", "ok");
  } catch (e) { toast(e.message, "err"); }
}

/* ============================================================
   CANVASES
   ============================================================ */
function prepCanvas(cv) {
  /* Layout size MUST be CSS-derived and locked: on displays with
     devicePixelRatio > 1, setting the bitmap (attribute) size of a canvas
     whose display size follows its attributes feeds back into clientWidth,
     and the canvas grows exponentially every draw ("ROI keeps zooming in,
     then everything goes white"). Locking style.width/height breaks the
     feedback loop permanently. */
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = cv.getBoundingClientRect();
  const w = Math.round(rect.width) || cv.width || 300;
  const h = Math.round(rect.height) || cv.height || 150;
  if (!cv.style.width && w > 10 && h > 10) {
    cv.style.width = w + "px";
    cv.style.height = h + "px";
  }
  const W = Math.min(Math.round(w * dpr), 4096), H = Math.min(Math.round(h * dpr), 4096);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(W / w, 0, 0, H / h, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}
/* window resizes: drop the locks so canvases re-measure at the new size */
window.addEventListener("resize", () => {
  ["#traj-canvas", "#donut-canvas", "#fps-canvas", "#risk-canvas", "#roi-canvas"].forEach(s => {
    const c = document.querySelector(s);
    if (c) { c.style.width = ""; c.style.height = ""; }
  });
});
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function drawDonut(s) {
  const cv = $("#donut-canvas");
  if (!cv || !cv.offsetParent) return;
  const [ctx, w, h] = prepCanvas(cv);
  const cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 8, r = R * 0.68;
  const dist = s.distribution || {};
  const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
  const total = s.counts.total || 0;
  ctx.lineWidth = R - r;
  if (!total) {
    ctx.strokeStyle = css("--border"); ctx.beginPath();
    ctx.arc(cx, cy, (R + r) / 2, 0, Math.PI * 2); ctx.stroke();
  } else {
    let a0 = -Math.PI / 2;
    entries.slice(0, 6).forEach(([k, v], i) => {
      const a1 = a0 + (v / total) * Math.PI * 2;
      ctx.strokeStyle = PALETTE[i % PALETTE.length];
      ctx.beginPath(); ctx.arc(cx, cy, (R + r) / 2, a0 + 0.02, a1 - 0.02); ctx.stroke();
      a0 = a1;
    });
  }
  $("#donut-total").textContent = total;
  $("#dist-legend").innerHTML = total
    ? entries.slice(0, 6).map(([k, v], i) => `
        <div class="lg-row"><span class="lg-left"><i class="sw" style="background:${PALETTE[i % PALETTE.length]}"></i>${k}</span><span class="lg-count mono">${v}</span></div>`).join("")
    : `<div class="empty-note">No objects in frame</div>`;
}

function drawSparkline(s) {
  const cv = $("#fps-canvas");
  if (!cv || !cv.offsetParent) return;
  const [ctx, w, h] = prepCanvas(cv);
  const data = s.fps_history || [];
  const accent = css("--accent"), txt = css("--text2");
  ctx.font = "9px ui-monospace, monospace"; ctx.fillStyle = txt;
  if (data.length < 2) { ctx.fillText("waiting for data…", 8, h / 2); return; }
  const max = Math.max(...data, 1), min = Math.min(...data);
  const X = i => 6 + (i / (data.length - 1)) * (w - 12);
  const Y = v => h - 8 - ((v - 0) / (max * 1.15)) * (h - 18);
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, accent + "44"); grad.addColorStop(1, accent + "00");
  ctx.beginPath(); ctx.moveTo(X(0), Y(data[0]));
  data.forEach((v, i) => ctx.lineTo(X(i), Y(v)));
  ctx.strokeStyle = accent; ctx.lineWidth = 1.6; ctx.stroke();
  ctx.lineTo(X(data.length - 1), h - 4); ctx.lineTo(X(0), h - 4); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  ctx.fillStyle = txt; ctx.fillText(`${fmt(max, 0)} fps`, w - 46, 12);
  ctx.fillText(`${fmt(data[data.length - 1], 1)}`, w - 44, h - 8);
}

function drawRiskStrip(s) {
  const cv = $("#risk-canvas");
  if (!cv || !cv.offsetParent) return;
  const [ctx, w, h] = prepCanvas(cv);
  const hist = s.risk_history || [];
  const colors = { SAFE: "#4ADE80", WARNING: "#FBBF24", DANGER: "#FB923C", CRITICAL: "#F87171" };
  const n = Math.max(hist.length, 1);
  const bw = w / 160;
  for (let i = 0; i < hist.length; i++) {
    ctx.fillStyle = colors[hist[i]] || "#4ADE80";
    const x = w - (hist.length - i) * bw;
    ctx.globalAlpha = 0.28;
    ctx.fillRect(x, 6, Math.ceil(bw), h - 12);
  }
  ctx.globalAlpha = 1;
  if (!hist.length) { ctx.font = "10px sans-serif"; ctx.fillStyle = css("--text2"); ctx.fillText("no samples yet", 8, h / 2 + 3); }
}

function polyMap(points, fw, fh, w, h) {
  const s = Math.min(w / fw, h / fh) * 0.94;
  const ox = (w - fw * s) / 2, oy = (h - fh * s) / 2;
  return points.map(([x, y]) => [ox + x * s, oy + y * s]);
}

function drawScene(ctx, s, w, h, { grid = false, labels = false } = {}) {
  const poly = s.roi_polygon || {};
  const staticPoly = s.static_roi_polygon || {};
  if (!poly.width && !staticPoly.width) {
    ctx.font = "10px sans-serif"; ctx.fillStyle = css("--text2"); ctx.fillText("no video", 8, 14); return;
  }
  const fw = poly.width || staticPoly.width, fh = poly.height || staticPoly.height;
  const accent = css("--accent"), bd = css("--border"), txt = css("--text2");
  const roiMeta = s.roi || poly;
  const overlays = s.config?.overlays || {};
  const dynamicState = roiMeta.state || poly.state;
  const showDynamic = overlays.roi !== false && !["fallback", "disabled"].includes(dynamicState);
  const showStatic = overlays.static_roi !== false;
  const uncertain = !!(roiMeta.uncertainty ?? poly.uncertainty);
  const roiColor = uncertain ? "#FBBF24" : "#4ADE80";
  const pts = polyMap(poly.points || [], fw, fh, w, h);
  const staticPts = polyMap(staticPoly.points || [], fw, fh, w, h);
  const toXY = (x, y) => polyMap([[x, y]], fw, fh, w, h)[0];

  if (grid) {
    ctx.strokeStyle = bd; ctx.lineWidth = 0.6;
    for (let gx = 0; gx <= 4; gx++) { const [x] = toXY(fw * gx / 4, 0); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let gy = 0; gy <= 3; gy++) { const [, y] = toXY(0, fh * gy / 3); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  }

  // Fixed DrivingROI trapezoid: this is the geometry edited by the static ROI sliders.
  if (showStatic && staticPts.length >= 3) {
    ctx.beginPath();
    staticPts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath();
    ctx.fillStyle = accent + "12"; ctx.fill();
    ctx.strokeStyle = accent; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([]);
  }

  if (showDynamic && pts.length >= 3) {
    ctx.beginPath();
    pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath();
    ctx.fillStyle = roiColor + "1C"; ctx.fill();
    ctx.strokeStyle = roiColor; ctx.lineWidth = 1.4; ctx.stroke();
  }

  const drawBoundary = points => {
    if (!points || points.length < 2) return;
    const mapped = polyMap(points, fw, fh, w, h);
    ctx.beginPath();
    mapped.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.strokeStyle = roiColor;
    ctx.lineWidth = 1.8;
    ctx.setLineDash(uncertain ? [4, 3] : []);
    ctx.stroke();
    ctx.setLineDash([]);
  };
  if (overlays.lane !== false) {
    drawBoundary(poly.left_boundary);
    drawBoundary(poly.right_boundary);
  }
  if (overlays.search_region && poly.search_region?.length >= 3) {
    const search = polyMap(poly.search_region, fw, fh, w, h);
    ctx.beginPath();
    search.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath(); ctx.strokeStyle = accent; ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]); ctx.stroke(); ctx.setLineDash([]);
  }

  if (labels && showStatic) {
    const corners = staticPoly.corners?.length ? staticPoly.corners : staticPoly.points?.slice(0, 4) || [];
    const cornerPts = polyMap(corners, fw, fh, w, h);
    ctx.font = "8.5px ui-monospace, monospace"; ctx.fillStyle = txt;
    ["TL", "TR", "BR", "BL"].forEach((t, i) => {
      if (cornerPts[i]) ctx.fillText(t, cornerPts[i][0] - 6, cornerPts[i][1] - 4);
    });
  }

  // ego marker at bottom-center of the configured fallback corridor
  const roiCfg = s.config?.roi || {};
  const [ebx, eby] = toXY(fw * (roiCfg.ego_center_ratio ?? 0.5), fh * (1 - (roiCfg.bottom_margin ?? 0)));
  ctx.fillStyle = accent;
  ctx.beginPath(); ctx.moveTo(ebx, eby - 7); ctx.lineTo(ebx - 5, eby + 4); ctx.lineTo(ebx + 5, eby + 4); ctx.closePath(); ctx.fill();

  // detections (all objects remain represented, including outside/uncertain)
  (s.detections || []).forEach(d => {
    const [x, y] = toXY((d.bbox[0] + d.bbox[2]) / 2, d.bbox[3]);
    ctx.fillStyle = d.roi_uncertainty ? "#FBBF24" : d.in_path
      ? ({ SAFE: "#4ADE80", WARNING: "#FBBF24", DANGER: "#FB923C", CRITICAL: "#F87171" }[d.risk] || "#4ADE80")
      : "#94A3B8";
    ctx.beginPath(); ctx.arc(x, y, 3.2, 0, Math.PI * 2); ctx.fill();
  });
}

function drawTrajectory(s) {
  const cv = $("#traj-canvas");
  if (!cv || !cv.offsetParent) return;
  const [ctx, w, h] = prepCanvas(cv);
  drawScene(ctx, s, w, h);
}

function drawRoiSchematic(s) {
  const cv = $("#roi-canvas");
  if (!cv || !cv.offsetParent) return;
  const [ctx, w, h] = prepCanvas(cv);
  drawScene(ctx, s, w, h, { grid: true, labels: true });
  ctx.font = "10px ui-monospace, monospace"; ctx.fillStyle = css("--text2");
  ctx.fillText("corridor from image-space geometry", 8, 14);
  const roi = s.roi || {};
  const state = (roi.state || "unavailable").replaceAll("_", " ").toUpperCase();
  ctx.fillStyle = roi.uncertainty ? "#FBBF24" : "#4ADE80";
  ctx.fillText(`${state} · confidence ${fmt(roi.confidence, 2)}`, 8, 28);
}

/* refresh canvases when theme flips */
new MutationObserver(() => { if (State.stats) { drawDonut(State.stats); drawSparkline(State.stats); drawRiskStrip(State.stats); drawTrajectory(State.stats); drawRoiSchematic(State.stats); } })
  .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

/* ============================================================
   BOOT
   ============================================================ */
document.addEventListener("DOMContentLoaded", () => {
  Theme.init();
  initClock();
  initRouter();
  initControls();
  initUploads();

  Video.init();
  console.log("[AutoSense] UI build 5 — device badge + runtime GPU/CPU switching");
  const sm = localStorage.getItem("autosense.stream") === "polling" ? "polling" : "mjpeg";
  setStreamMode(sm, false);

  // boot re-apply: the backend keeps imgsz in memory only, so restore Turbo
  // after a page reload / server restart if the user left it enabled
  if (localStorage.getItem("autosense.turbo") === "1") applyTurbo(true, false);

  startWS();
  // initial snapshot so UI isn't blank pre-WS
  api("/api/status").then(onStats).catch(() => {});
  loadSources();

  // connection age ticker
  setInterval(() => {
    if (State.lastUpdate) $("#conn-age").textContent = `${((Date.now() - State.lastUpdate) / 1000).toFixed(0)}s ago`;
  }, 1000);
});
})();
