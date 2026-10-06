/* Guided walkthrough shared by both pages.
   Tour.start(steps, from) — steps: [{sel?, title, body, before?}]. A step without `sel` (or whose
   element is missing) shows as a centred card. The target is spotlighted (everything else dimmed);
   Back / Next / Skip, ← → Esc keys. Auto-start once per page via `Tour.autoStart(...)` (remembered in
   localStorage, wrapped in try/catch so private windows still work). */
(function () {
  const css = `
  .tour-shade{position:fixed;inset:0;z-index:9998;pointer-events:auto}
  .tour-spot{position:fixed;z-index:9999;border-radius:10px;pointer-events:none;
    box-shadow:0 0 0 3px var(--accent,#5fc3e4),0 0 0 9999px rgba(3,6,10,.62);transition:all .25s ease}
  .tour-card{position:fixed;z-index:10000;width:min(380px,calc(100vw - 32px));background:var(--bg-elev,#121821);
    color:var(--fg,#e8edf3);border:1px solid var(--accent,#5fc3e4);border-radius:12px;padding:14px 16px 12px;
    box-shadow:0 18px 50px rgba(0,0,0,.45);font-family:var(--sans,system-ui);font-size:13px;line-height:1.5}
  .tour-card .tc-step{font-size:10.5px;letter-spacing:.5px;text-transform:uppercase;color:var(--accent,#5fc3e4);font-weight:600}
  .tour-card h3{margin:3px 0 6px;font-size:15.5px}
  .tour-card p{margin:0 0 6px}
  .tour-card ul{margin:2px 0 6px;padding-left:18px}
  .tour-card .tc-bar{display:flex;gap:6px;align-items:center;margin-top:10px}
  .tour-card .tc-dots{display:flex;gap:4px;margin-right:auto}
  .tour-card .tc-dots i{width:6px;height:6px;border-radius:50%;background:var(--border,#2a3547)}
  .tour-card .tc-dots i.on{background:var(--accent,#5fc3e4)}
  .tour-card button{font:inherit;border:1px solid var(--border,#2a3547);background:var(--surface,#171f2b);color:inherit;
    border-radius:7px;padding:5px 12px;cursor:pointer;font-weight:600}
  .tour-card button.primary{background:var(--accent,#5fc3e4);color:#06131a;border-color:var(--accent,#5fc3e4)}
  .tour-card button.link{background:none;border:none;color:var(--fg-muted,#8a96a8);font-weight:500;padding:5px 4px}
  `;
  const style = document.createElement("style"); style.textContent = css; document.head.appendChild(style);

  let steps = [], i = 0, shade, spot, card, onKey, timer;

  function close() {
    [shade, spot, card].forEach(e => e && e.remove()); shade = spot = card = null; clearInterval(timer);
    document.removeEventListener("keydown", onKey, true); window.removeEventListener("resize", place);
  }

  function place() {
    if (!card) return;
    const s = steps[i], target = s.sel ? document.querySelector(s.sel) : null;
    const vw = window.innerWidth, vh = window.innerHeight, cw = card.offsetWidth, ch = card.offsetHeight, pad = 8;
    if (!target || !target.getClientRects().length) {
      spot.style.cssText = `left:${vw / 2}px;top:${vh / 2}px;width:0;height:0`;
      card.style.left = (vw - cw) / 2 + "px"; card.style.top = Math.max(16, (vh - ch) / 2) + "px"; return;
    }
    let r = target.getBoundingClientRect();
    if (r.bottom < 40 || r.top > vh - 40) {   // content above loaded late and pushed the target away: bring it back
      target.scrollIntoView({block: r.height > vh * .8 ? "start" : "center", behavior: "instant"}); r = target.getBoundingClientRect();
    }
    const top = Math.max(4, r.top - pad), left = Math.max(4, r.left - pad);
    const w = Math.min(vw - 8, r.width + 2 * pad), h = Math.min(vh - 8, r.bottom + pad) - top;
    spot.style.cssText = `left:${left}px;top:${top}px;width:${w}px;height:${Math.max(h, 20)}px`;
    let cy = r.bottom + 14;                                  // below the target if it fits…
    if (cy + ch > vh - 12) cy = r.top - ch - 14;             // …else above…
    if (cy < 12) cy = Math.min(vh - ch - 12, Math.max(12, r.top + 12));  // …else inside, near its top
    cy = Math.min(Math.max(12, cy), Math.max(12, vh - ch - 12));   // never off-screen
    let cx = Math.min(vw - cw - 12, Math.max(12, r.left));
    card.style.left = cx + "px"; card.style.top = cy + "px";
  }

  async function show(n) {
    i = Math.max(0, Math.min(steps.length - 1, n));
    const s = steps[i];
    if (s.before) { try { await s.before(); } catch (e) { console.warn(e); } }
    const target = s.sel ? document.querySelector(s.sel) : null;
    if (target) target.scrollIntoView({block: "center", behavior: "instant"});
    card.innerHTML = `<div class="tc-step">Step ${i + 1} of ${steps.length}</div><h3>${s.title}</h3>${s.body}
      <div class="tc-bar"><span class="tc-dots">${steps.map((_, k) => `<i class="${k === i ? "on" : ""}"></i>`).join("")}</span>
      <button class="link" data-a="skip">Skip tour</button>
      ${i ? `<button data-a="back">Back</button>` : ""}
      <button class="primary" data-a="next">${i === steps.length - 1 ? "Finish" : "Next →"}</button></div>`;
    card.querySelector('[data-a="next"]').focus();
    requestAnimationFrame(place);
  }

  function start(list, from = 0) {
    close(); steps = list;
    shade = document.createElement("div"); shade.className = "tour-shade";
    spot = document.createElement("div"); spot.className = "tour-spot";
    card = document.createElement("div"); card.className = "tour-card"; card.setAttribute("role", "dialog"); card.setAttribute("aria-live", "polite");
    document.body.append(shade, spot, card);
    card.addEventListener("click", e => { const a = e.target.closest("button")?.dataset.a;
      if (a === "next") (i === steps.length - 1 ? close() : show(i + 1)); if (a === "back") show(i - 1); if (a === "skip") close(); });
    shade.addEventListener("click", () => show(i + 1 < steps.length ? i + 1 : i));
    onKey = e => { if (e.key === "Escape") close(); if (e.key === "ArrowRight") { e.stopPropagation(); i === steps.length - 1 ? close() : show(i + 1); }
      if (e.key === "ArrowLeft") { e.stopPropagation(); show(i - 1); } };
    document.addEventListener("keydown", onKey, true);
    window.addEventListener("resize", place);
    timer = setInterval(place, 500);   // targets can grow (e.g. an answer arriving) — keep the spotlight on them
    show(from);
  }

  function autoStart(list, key) {
    let seen = false;
    try { seen = localStorage.getItem("tour:" + key) === "1"; localStorage.setItem("tour:" + key, "1"); } catch (e) { /* storage blocked */ }
    const param = new URLSearchParams(location.search).get("tour");
    if (param === "off") return;                                                     // ?tour=off: never (screenshots)
    const forced = parseInt(param, 10);                                              // ?tour=N opens it at step N (demo links)
    if (forced > 0) return start(list, forced - 1);
    if (!seen) start(list);
  }

  window.Tour = {start, autoStart, close};
})();
