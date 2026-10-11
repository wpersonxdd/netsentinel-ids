// NetSentinel motion video runtime: icons, background, cue timing, counters, captions and frame seeking.
// The renderer (scripts/make_media.py motion) injects window.CUES (start of each narration line, seconds),
// window.LDUR (length of each line), window.LINES (caption text) and window.DURATION before the page loads.
// Opening a scene file directly in a browser plays it with default timings.
(function () {
  const ICONS = {
    shield: '<path d="M24 4 L40 10 V22 C40 33 33 40 24 44 C15 40 8 33 8 22 V10 Z"/>',
    logo: '<path d="M24 4 L40 10 V22 C40 33 33 40 24 44 C15 40 8 33 8 22 V10 Z"/><path d="M13 25 H18 L21 18 L25 31 L28 23 H35"/>',
    check: '<path d="M10 25 L20 35 L38 14"/>',
    cross: '<path d="M13 13 L35 35 M35 13 L13 35"/>',
    database: '<ellipse cx="24" cy="11" rx="14" ry="5"/><path d="M10 11 V37 C10 40 16 42 24 42 C32 42 38 40 38 37 V11"/><path d="M10 24 C10 27 16 29 24 29 C32 29 38 27 38 24"/>',
    funnel: '<path d="M6 8 H42 L28 26 V40 L20 44 V26 Z"/>',
    neural: '<path d="M10 14 L24 10 M10 14 L24 24 M10 14 L24 38 M10 34 L24 10 M10 34 L24 24 M10 34 L24 38 M24 10 L38 24 M24 24 L38 24 M24 38 L38 24"/><circle cx="10" cy="14" r="4"/><circle cx="10" cy="34" r="4"/><circle cx="24" cy="10" r="4"/><circle cx="24" cy="24" r="4"/><circle cx="24" cy="38" r="4"/><circle cx="38" cy="24" r="4"/>',
    radar: '<circle cx="24" cy="24" r="19"/><circle cx="24" cy="24" r="12"/><circle cx="24" cy="24" r="5"/><path d="M24 24 L38 11"/><circle cx="33" cy="31" r="2"/>',
    scales: '<path d="M24 6 V40 M15 42 H33 M8 13 H40"/><path d="M8 13 L3 26 H13 Z"/><path d="M40 13 L35 26 H45 Z"/>',
    wave: '<path d="M4 30 C10 30 12 14 18 14 C24 14 24 34 30 34 C36 34 38 20 44 20"/><path d="M4 42 H44"/>',
    bell: '<path d="M12 34 V22 C12 15 17 10 24 10 C31 10 36 15 36 22 V34 L40 38 H8 Z"/><path d="M20 42 C21 44.5 27 44.5 28 42"/><path d="M24 6 V10"/>',
    ticket: '<rect x="6" y="10" width="36" height="28" rx="4"/><path d="M12 18 H26 M12 24 H36 M12 30 H22"/><circle cx="35" cy="18" r="2.5"/>',
    person: '<circle cx="24" cy="15" r="7"/><path d="M10 42 C10 32 16 27 24 27 C32 27 38 32 38 42"/>',
    guard: '<circle cx="24" cy="12" r="6"/><path d="M9 44 C9 32 15 23 24 23 C33 23 39 32 39 44"/><path d="M24 29 L31 31.5 V36 C31 39.5 28 42 24 43.5 C20 42 17 39.5 17 36 V31.5 Z"/>',
    loop: '<path d="M38 20 A15 15 0 0 0 10 18"/><path d="M10 28 A15 15 0 0 0 38 30"/><path d="M38 9 V20 H27"/><path d="M10 39 V28 H21"/>',
    magnifier: '<circle cx="20" cy="20" r="12"/><path d="M29 29 L42 42"/>',
    clock: '<circle cx="24" cy="24" r="18"/><path d="M24 13 V24 L32 29"/>',
    calendar: '<rect x="6" y="10" width="36" height="32" rx="4"/><path d="M6 19 H42 M16 5 V13 M32 5 V13"/>',
    bug: '<ellipse cx="24" cy="28" rx="9" ry="12"/><circle cx="24" cy="13" r="4"/><path d="M24 20 V40 M15 23 H6 M15 31 H6 M33 23 H42 M33 31 H42 M20 10 L16 5 M28 10 L32 5"/>',
    ghost: '<path d="M12 42 V20 C12 13 17 8 24 8 C31 8 36 13 36 20 V42 L31 38 L27 42 L24 38 L21 42 L17 38 Z"/><circle cx="19" cy="21" r="2"/><circle cx="29" cy="21" r="2"/>',
    warning: '<path d="M24 6 L44 40 H4 Z"/><path d="M24 18 V28"/><circle cx="24" cy="34" r="1"/>',
    monitor: '<rect x="5" y="8" width="38" height="26" rx="3"/><path d="M18 42 H30 M24 34 V42"/>',
    code: '<path d="M16 14 L6 24 L16 34 M32 14 L42 24 L32 34 M27 10 L21 38"/>',
    flask: '<path d="M19 6 H29 M21 6 V18 L9 38 C8 41 10 43 13 43 H35 C38 43 40 41 39 38 L27 18 V6"/><path d="M14 30 H34"/>',
    interval: '<path d="M6 24 H42"/><path d="M11 15 V33 M37 15 V33"/><circle cx="24" cy="24" r="5"/>',
    splits: '<path d="M6 12 H26 M31 12 H42 M6 24 H16 M21 24 H42 M6 36 H34 M39 36 H42"/>',
    globe: '<circle cx="24" cy="24" r="18"/><path d="M6 24 H42 M24 6 C30 12 30 36 24 42 C18 36 18 12 24 6"/>',
    eye: '<path d="M4 24 C10 14 17 10 24 10 C31 10 38 14 44 24 C38 34 31 38 24 38 C17 38 10 34 4 24 Z"/><circle cx="24" cy="24" r="6"/>',
    sliders: '<path d="M8 14 H40 M8 24 H40 M8 34 H40"/><circle cx="18" cy="14" r="3.5"/><circle cx="32" cy="24" r="3.5"/><circle cx="14" cy="34" r="3.5"/>',
    terminal: '<rect x="5" y="9" width="38" height="30" rx="3"/><path d="M12 19 L18 24 L12 29 M22 30 H32"/>',
    rows: '<rect x="6" y="8" width="36" height="32" rx="3"/><path d="M6 18 H42 M6 28 H42 M18 8 V40"/>',
    copies: '<rect x="14" y="14" width="26" height="26" rx="3"/><path d="M8 32 V10 C8 9 9 8 10 8 H32"/>',
    target: '<circle cx="24" cy="24" r="18"/><circle cx="24" cy="24" r="10"/><circle cx="24" cy="24" r="2.5"/>',
    arrow: '<path d="M6 24 H40 M30 14 L40 24 L30 34"/>',
    doc: '<path d="M12 5 H30 L38 13 V43 H12 Z"/><path d="M30 5 V13 H38 M18 22 H32 M18 29 H32 M18 36 H26"/>',
    lock: '<rect x="10" y="21" width="28" height="21" rx="3"/><path d="M16 21 V15 C16 10 19.5 7 24 7 C28.5 7 32 10 32 15 V21"/>',
    packet: '<rect x="8" y="12" width="32" height="24" rx="4"/><path d="M8 20 H40"/>',
  };

  function iconSVG(name) {
    const body = (ICONS[name] || ICONS.target).replace(/<(path|circle|ellipse|rect|line|polyline)/g, '<$1 pathLength="1"');
    return '<svg class="ic" viewBox="0 0 48 48">' + body + '</svg>';
  }
  window.iconSVG = iconSVG;

  // deterministic random numbers, so every render of a scene is identical
  let seed = 7;
  const rnd = window.rnd = function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };

  // background
  const bg = document.createElement('div');
  bg.className = 'bg';
  let html = '<div class="grid"></div>' +
    '<div class="blob" style="left:-200px;top:-260px;width:900px;height:700px;background:#1C4FD8"></div>' +
    '<div class="blob" style="right:-260px;bottom:-300px;width:900px;height:700px;background:#7B2CBF;animation-delay:-4s"></div>';
  for (let i = 0; i < 70; i++) {
    html += '<i class="p" style="left:' + (rnd() * 1920).toFixed(0) + 'px;top:' + (rnd() * 1080).toFixed(0) +
      'px;animation-delay:-' + (rnd() * 4).toFixed(2) + 's;animation-duration:' + (3 + rnd() * 4).toFixed(2) + 's;' +
      (rnd() < .25 ? 'background:#FF4FD8;box-shadow:0 0 8px #FF4FD8' : '') + '"></i>';
  }
  bg.innerHTML = html + '<div class="vignette"></div>';
  document.body.prepend(bg);

  // cue timing -> CSS variables --c0, --c1, ... and --end
  const CUES = window.CUES || [0.6, 4, 8, 12, 16, 20, 24];
  const DUR = window.DURATION || (CUES[CUES.length - 1] + 5);
  const LDUR = window.LDUR || CUES.map((c, i) => (CUES[i + 1] || DUR) - c - 0.4);
  const root = document.documentElement;
  for (let i = 0; i < 12; i++) root.style.setProperty('--c' + i, (i < CUES.length ? CUES[i] : DUR + 10) + 's');
  root.style.setProperty('--end', DUR + 's');
  const cue = window.cueTime = i => (i < CUES.length ? CUES[i] : DUR + 10);

  const hooks = [];
  window.onFrame = fn => hooks.push(fn);
  window.seek = function (t) {
    document.getAnimations().forEach(a => { a.pause(); a.currentTime = t * 1000; });
    hooks.forEach(fn => fn(t));
  };

  // camera move over a w x h layer (screenshot + overlays): keys [{t, s, x, y}] = at time t, zoom s centred on (x, y)
  window.camera = function (el, w, h, keys, d) {
    d = d || 1.2;
    el.style.transformOrigin = '0 0';
    const ease = p => (p < .5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);
    const place = k => {
      const tx = Math.min(0, Math.max(w - w * k.s, w / 2 - k.x * k.s));
      const ty = Math.min(0, Math.max(h - h * k.s, h / 2 - k.y * k.s));
      el.style.transform = 'translate(' + tx.toFixed(2) + 'px,' + ty.toFixed(2) + 'px) scale(' + k.s.toFixed(4) + ')';
    };
    onFrame(t => {
      let i = -1;
      keys.forEach((k, j) => { if (k.t <= t) i = j; });
      if (i <= 0) return place(keys[0]);
      const p = Math.min(1, (t - keys[i].t) / d), e = ease(p), a = keys[i - 1], b = keys[i];
      place({ s: a.s + (b.s - a.s) * e, x: a.x + (b.x - a.x) * e, y: a.y + (b.y - a.y) * e });
    });
  };

  // scene scripts placed after this file add their elements first; then icons, counters, captions are wired up
  document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('[data-icon]').forEach(el => { el.innerHTML = iconSVG(el.dataset.icon); });

  // counters: <span data-count="22601" data-cue="2" data-off="0.2" data-dur="1.5" data-dec="0" data-from="0" data-suf="%">
  document.querySelectorAll('[data-count]').forEach(el => {
    const to = +el.dataset.count, from = +(el.dataset.from || 0), dec = +(el.dataset.dec || 0);
    const at = cue(+(el.dataset.cue || 0)) + +(el.dataset.off || 0), d = +(el.dataset.dur || 1.4);
    const pre = el.dataset.pre || '', suf = el.dataset.suf || '';
    const fmt = v => v.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
    onFrame(t => {
      const p = Math.min(1, Math.max(0, (t - at) / d)), e = 1 - Math.pow(1 - p, 3);
      el.textContent = pre + fmt(from + (to - from) * e) + suf;
    });
  });

  // typed text: <span data-type="python -m nids all" data-cue="4" data-off="0.5" data-cps="18">
  document.querySelectorAll('[data-type]').forEach(el => {
    const txt = el.dataset.type, at = cue(+(el.dataset.cue || 0)) + +(el.dataset.off || 0), cps = +(el.dataset.cps || 20);
    onFrame(t => { el.textContent = txt.slice(0, Math.max(0, Math.floor((t - at) * cps))); });
  });

  // burned-in captions
  if (window.LINES && window.SHOW_CAPTIONS !== false) {
    const cap = document.createElement('div');
    cap.className = 'cap';
    document.body.appendChild(cap);
    onFrame(t => {
      let txt = '';
      CUES.forEach((c, i) => { if (t >= c && t < c + LDUR[i] + 0.25) txt = window.LINES[i]; });
      if (cap.textContent !== txt) cap.textContent = txt;
    });
  }

  // fade in from / out to black
  const veil = document.createElement('div');
  veil.className = 'veil';
  document.body.appendChild(veil);
  veil.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 500, fill: 'both' });
  veil.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 450, delay: (DUR - 0.45) * 1000, fill: 'forwards' });

  // the renderer calls seek(t) before every frame; in a normal browser the page simply plays
  if (!window.CAPTURE) {
    const t0 = performance.now();
    const tick = () => { hooks.forEach(fn => fn((performance.now() - t0) / 1000)); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  });
})();
