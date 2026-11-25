/*
  Readiness front end: the International View and the Bilateral View.

  International View: every country coloured by its readiness band (1 = most
  difficult conditions, 5 = best placed), with its criteria and the reasons
  it sits where it does.

  Bilateral View: click a country and the map shows every other country from
  that country's point of view, coloured by how cooperative or conflictual
  their recent contact has been (relation band 1 = most conflictual,
  5 = most cooperative). Countries with too little contact to judge are
  hatched, never coloured as good relations (decision log 25.3).

  Data comes from python -m src.export_web (see index.html for how to serve
  it). Colours always come with words: the legend, the tooltip and the panels
  say what each colour means (decision log 21.6).
*/
(function () {
  const INK = '#1b2230';
  const DATA = '../../data/web/';
  const MAX_ZOOM = 120;

  const BAND_COLOURS = {
    1: { bg: '#9c3a12', fg: '#ffffff' }, 2: { bg: '#e07b39', fg: INK },
    3: { bg: '#f3dcae', fg: INK }, 4: { bg: '#9cc3e0', fg: INK },
    5: { bg: '#2c6694', fg: '#ffffff' }
  };
  const RELATION_COLOURS = {
    1: { bg: '#b2182b', fg: '#ffffff' }, 2: { bg: '#ef8a62', fg: INK },
    3: { bg: '#f2e6c4', fg: INK }, 4: { bg: '#a6d96a', fg: INK },
    5: { bg: '#1a9641', fg: '#ffffff' }
  };
  const HATCH = 'url(#nodata-hatch)';
  const HATCH_CSS = 'repeating-linear-gradient(45deg, #e3e5e8 0 5px, #bfc5cc 5px 7px)';

  // Full names in place of the map data's abbreviations
  const FIX = {
    'eSwatini': 'Eswatini', 'Macedonia': 'North Macedonia', 'Turkey': 'Türkiye',
    'United States of America': 'United States', 'S. Sudan': 'South Sudan',
    'Dem. Rep. Congo': 'DR Congo', 'Central African Rep.': 'Central African Republic',
    'Dominican Rep.': 'Dominican Republic', 'Eq. Guinea': 'Equatorial Guinea',
    'Bosnia and Herz.': 'Bosnia and Herzegovina', 'Solomon Is.': 'Solomon Islands',
    'Marshall Is.': 'Marshall Islands', 'Antigua and Barb.': 'Antigua and Barbuda',
    'St. Vin. and Gren.': 'Saint Vincent and the Grenadines', 'St. Kitts and Nevis': 'Saint Kitts and Nevis',
    'W. Sahara': 'Western Sahara', 'N. Cyprus': 'Northern Cyprus', 'Fr. Polynesia': 'French Polynesia',
    'Fr. S. Antarctic Lands': 'French Southern Territories', 'N. Mariana Is.': 'Northern Mariana Islands',
    'U.S. Virgin Is.': 'US Virgin Islands', 'British Virgin Is.': 'British Virgin Islands',
    'S. Geo. and the Is.': 'South Georgia and the South Sandwich Islands',
    'Br. Indian Ocean Ter.': 'British Indian Ocean Territory', 'Indian Ocean Ter.': 'Australian Indian Ocean Territories',
    'Pitcairn Is.': 'Pitcairn Islands', 'Falkland Is.': 'Falkland Islands', 'Cayman Is.': 'Cayman Islands',
    'Turks and Caicos Is.': 'Turks and Caicos Islands', 'Cook Is.': 'Cook Islands',
    'St. Pierre and Miquelon': 'Saint Pierre and Miquelon', 'Wallis and Futuna Is.': 'Wallis and Futuna',
    'Faeroe Is.': 'Faroe Islands', 'Heard I. and McDonald Is.': 'Heard Island and McDonald Islands',
    'Ashmore and Cartier Is.': 'Ashmore and Cartier Islands'
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const pct = (x) => (Math.max(0, Math.min(1, x)) * 100).toFixed(1) + '%';
  const items = (list) => '<ul class="items">' + list.map((x) => '<li>' + esc(x) + '</li>').join('') + '</ul>';
  const longDate = (iso) => new Date(iso + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });

  async function getJSON(url) {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(url + ': ' + response.status);
    return response.json();
  }

  // ---- State ----
  const state = { view: 'international', date: null, sel: 'BRA', intl: null, bil: null };
  let view = { x: 0, y: 0, w: 960 };
  let drag = null, wasDrag = false;
  let W = 960, H = 470;
  let names = {}, keys = [], biggest = {}, pathsByKey = {};

  const mapEl = $('map'), svg = $('world'), group = $('countries'), tooltip = $('tooltip');

  // ---- Map: draw every country once ----
  function buildMap(geo) {
    W = geo.w; H = geo.h; view = { x: 0, y: 0, w: W };
    const SVGNS = 'http://www.w3.org/2000/svg';
    for (const f of geo.countries) {
      const key = f.iso || f.name;
      const name = FIX[f.name] || f.name;
      if (!names[key]) { names[key] = name; keys.push(key); }
      if (!biggest[key] || f.area > biggest[key].area) biggest[key] = f;
      const p = document.createElementNS(SVGNS, 'path');
      p.setAttribute('d', f.d);
      p.dataset.key = key;
      group.appendChild(p);
      (pathsByKey[key] = pathsByKey[key] || []).push(p);
    }
    keys.sort((a, b) => names[a].localeCompare(names[b]));
    $('find-country').innerHTML = keys.map((k) => '<option value="' + esc(k) + '">' + esc(names[k]) + '</option>').join('');
  }

  // ---- What each country looks like in the current view ----
  function international(key) {
    return state.intl && state.intl.countries[key];
  }

  function partners() {
    return (state.bil && state.bil.partners[state.sel]) || {};
  }

  function relationBand(band) {
    return state.bil ? state.bil.bands.find((b) => b.band === band) : null;
  }

  function fillFor(key) {
    if (state.view === 'international') {
      const c = international(key);
      return c ? BAND_COLOURS[c.band].bg : HATCH;
    }
    if (key === state.sel) return INK;
    const p = partners()[key];
    return p && p[0] > 0 ? RELATION_COLOURS[p[0]].bg : HATCH;
  }

  function tipFor(key) {
    const name = esc(names[key] || key);
    if (state.view === 'international') {
      const c = international(key);
      if (!c) return name + ': not scored on this date';
      const band = state.intl.bands.find((b) => b.band === c.band);
      return name + ': band ' + c.band + ', ' + esc(band.label) + ' (score ' + c.score.toFixed(2) + ')';
    }
    if (key === state.sel) return name + ' (selected)';
    const p = partners()[key];
    const days = state.bil ? state.bil.window_days : 90;
    if (!p) return name + ': no recorded contact with ' + esc(names[state.sel]) + ' in ' + days + ' days';
    const label = relationBand(p[0]).label;
    const share = p[2] ? Math.round(100 * p[3] / p[2]) : 0;
    return name + ': ' + esc(label) + '<br>' + p[2] + ' events in ' + days + ' days, ' + share + '% conflictual';
  }

  function recolour() {
    for (const key of keys) {
      const fill = fillFor(key);
      for (const p of pathsByKey[key]) p.setAttribute('fill', fill);
    }
  }

  // ---- Zoom, pan and labels ----
  function applyView() {
    const vh = view.w * H / W;
    svg.setAttribute('viewBox', [view.x, view.y, view.w, vh].map((v) => v.toFixed(3)).join(' '));
    const zoom = W / view.w;
    $('nodata-hatch').setAttribute('patternTransform', 'rotate(45) scale(' + (1 / Math.sqrt(zoom)).toFixed(4) + ')');
    $('zoom-label').textContent = zoom.toFixed(1) + '× zoom';
    drawLabels();
  }

  // Country names appear once zoomed in, for countries wide enough on screen
  function drawLabels() {
    const zoom = W / view.w, vh = view.w * H / W;
    const ppu = (mapEl.clientWidth || 900) / view.w;
    let html = '';
    if (zoom >= 1.2) {
      for (const key of keys) {
        const f = biggest[key], name = names[key];
        const need = Math.min(name.length * 6.5, 56);
        const fits = f.bw * ppu >= need || (f.bh * ppu >= need * 1.3 && f.bw * ppu >= 14);
        const inView = f.lx > view.x && f.lx < view.x + view.w && f.ly > view.y && f.ly < view.y + vh;
        if (!inView || !(fits || key === state.sel)) continue;
        html += '<span class="' + (key === state.sel ? 'sel' : '') + '" style="left:' +
          ((f.lx - view.x) / view.w * 100).toFixed(3) + '%;top:' + ((f.ly - view.y) / vh * 100).toFixed(3) + '%">' + esc(name) + '</span>';
      }
    }
    $('labels').innerHTML = html;
  }

  function setView(x, y, w) {
    w = Math.max(W / MAX_ZOOM, Math.min(W, w));
    const h = w * H / W;
    view = { w, x: Math.max(0, Math.min(W - w, x)), y: Math.max(0, Math.min(H - h, y)) };
    applyView();
  }

  function zoomAt(fx, fy, factor) {
    const vh = view.w * H / W;
    const px = view.x + fx * view.w, py = view.y + fy * vh;
    const nw = Math.max(W / MAX_ZOOM, Math.min(W, view.w / factor));
    setView(px - fx * nw, py - fy * (nw * H / W), nw);
  }

  mapEl.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = mapEl.getBoundingClientRect();
    zoomAt((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height, e.deltaY < 0 ? 1.35 : 1 / 1.35);
  }, { passive: false });

  svg.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false };
    wasDrag = false;
  });
  svg.addEventListener('pointermove', (e) => {
    const p = e.target.closest('path[data-key]');
    const r = mapEl.getBoundingClientRect();
    if (p && !(drag && drag.moved)) {
      tooltip.innerHTML = tipFor(p.dataset.key);
      tooltip.classList.remove('hidden');
      const x = e.clientX - r.left + 14, y = e.clientY - r.top + 14;
      tooltip.style.left = Math.min(x, r.width - 270) + 'px';
      tooltip.style.top = Math.min(y, r.height - 60) + 'px';
    } else {
      tooltip.classList.add('hidden');
    }
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; svg.classList.add('dragging'); }
    if (!drag.moved) return;
    setView(drag.vx - dx * view.w / r.width, drag.vy - dy * view.w / r.width, view.w);
  });
  const endDrag = () => { wasDrag = !!(drag && drag.moved); drag = null; svg.classList.remove('dragging'); };
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointerleave', () => { drag = null; svg.classList.remove('dragging'); tooltip.classList.add('hidden'); });
  svg.addEventListener('click', (e) => {
    if (wasDrag) { wasDrag = false; return; }
    const p = e.target.closest('path[data-key]');
    if (p) select(p.dataset.key);
  });

  $('zoom-in').addEventListener('click', () => zoomAt(0.5, 0.5, 1.5));
  $('zoom-out').addEventListener('click', () => zoomAt(0.5, 0.5, 1 / 1.5));
  $('zoom-reset').addEventListener('click', () => setView(0, 0, W));
  window.addEventListener('resize', drawLabels);
  $('find-country').addEventListener('change', (e) => select(e.target.value));

  // ---- Legend and distribution ----
  function drawLegend() {
    let html = '', counts = [];
    if (state.view === 'international') {
      const bands = state.intl ? state.intl.bands : [];
      html = bands.map((b) => '<div><span class="sw" style="background:' + BAND_COLOURS[b.band].bg + '"></span><span><strong>Band ' +
        b.band + '</strong> <span class="muted">' + esc(b.label) + '</span></span></div>').join('') +
        '<div><span class="sw hatch"></span><span class="muted">Not scored</span></div>';
      const scored = state.intl ? Object.values(state.intl.countries) : [];
      counts = [1, 2, 3, 4, 5].map((b) => [b, scored.filter((c) => c.band === b).length]);
      counts.push([0, keys.length - scored.length]);
      $('dist-label').textContent = 'Countries per band on this date';
    } else {
      const bands = state.bil ? state.bil.bands.filter((b) => b.band > 0) : [];
      html = '<div><span class="sw" style="background:' + INK + '"></span><span class="muted">Selected country</span></div>' +
        bands.map((b) => '<div><span class="sw" style="background:' + RELATION_COLOURS[b.band].bg + '"></span><span><strong>' +
          esc(b.label) + '</strong></span></div>').join('') +
        '<div><span class="sw hatch"></span><span class="muted">Too little contact to judge</span></div>';
      const list = Object.values(partners());
      counts = [1, 2, 3, 4, 5].map((b) => [b, list.filter((p) => p[0] === b).length]);
      counts.push([0, list.filter((p) => p[0] === 0).length]);
      $('dist-label').textContent = 'Countries with contact, by relation band';
    }
    $('legend').innerHTML = html;
    const colours = state.view === 'international' ? BAND_COLOURS : RELATION_COLOURS;
    $('dist').innerHTML = counts.filter(([, n]) => n > 0).map(([b, n]) => {
      const bg = b ? colours[b].bg : HATCH_CSS;
      const what = b ? (state.view === 'international' ? 'Band ' + b : relationBand(b).label) :
        (state.view === 'international' ? 'Not scored' : 'Too little contact');
      return '<div title="' + esc(what) + ': ' + n + '" style="flex:' + n + ' 1 0;background:' + bg +
        ';color:' + (b ? colours[b].fg : '#3d4654') + '">' + n + '</div>';
    }).join('');
  }

  // ---- International View panels ----
  function drawInternationalDetail() {
    const key = state.sel, name = names[key] || key;
    const c = international(key);
    let html = '<div class="detail-head"><div><div class="iso">' + esc(key) + '</div><h2>' + esc(name) + '</h2></div>';
    if (!c) {
      const missing = state.intl && state.intl.partial[key];
      const byKey = {};
      if (state.intl) for (const cr of state.intl.criteria) byKey[cr.key] = cr.name;
      html += '<div class="badge" style="background:#e3e5e8;color:#3d4654;font-size:13px;font-weight:600">No band</div></div>' +
        '<p class="note">' + (missing
          ? 'Not scored: no data on this date for ' + esc(missing.map((m) => byKey[m] || m).join(', ')) + '.'
          : 'Not scored: no data on this date.') + '</p>';
      $('detail').innerHTML = html;
      return;
    }
    const band = state.intl.bands.find((b) => b.band === c.band);
    const colour = BAND_COLOURS[c.band];
    const cuts = state.intl.cuts, edges = [0].concat(cuts, [1]);
    const err = 1.96 * c.se;
    const criteria = {};
    for (const cr of state.intl.criteria) criteria[cr.key] = cr;

    html += '<div class="badge" style="background:' + colour.bg + ';color:' + colour.fg + '">Band ' + c.band + '</div></div>' +
      '<div><div class="band-name">' + esc(band.label) + '</div>' + items(band.actions) + '</div>';
    if (c.drivers.length) {
      html += '<div class="field"><span class="field-label">Main reasons</span><ul class="items">' +
        c.drivers.map((k) => '<li><strong>' + esc(criteria[k].label) + ':</strong> ' + esc(criteria[k].text) + '</li>').join('') +
        '</ul></div>';
    }
    html += '<div class="field"><div class="score-row"><span class="field-label">Score and band range</span>' +
      '<span style="font-size:13px"><strong>' + c.score.toFixed(2) + '</strong> <span style="color:#5b6472">± ' + err.toFixed(2) +
      (c.band_low === c.band_high ? ' · band ' + c.band_low + ' only' : ' · bands ' + c.band_low + '–' + c.band_high) +
      '</span></span></div><div class="track-wrap"><div class="track">' +
      [1, 2, 3, 4, 5].map((b) => '<div style="background:' + BAND_COLOURS[b].bg + ';width:' + pct(edges[b] - edges[b - 1]) + '"></div>').join('') +
      '</div><div class="whisker" style="left:' + pct(c.score - err) + ';width:' + pct(2 * err) + '"></div>' +
      '<div class="marker" style="left:' + pct(c.score) + '"></div></div></div>' +
      '<div class="criteria"><span class="field-label">Criteria (0 to 1, 1 = best)</span>' +
      state.intl.groups.map((g) => {
        const rows = state.intl.criteria.filter((cr) => cr.group === g.key);
        if (!rows.length) return '';
        return '<div class="group"><h3>' + esc(g.name) + '</h3>' + rows.map((cr) => {
          const v = c.values[cr.key];
          return '<div class="crow"><span>' + esc(cr.name) + '</span><div class="bar"><div style="width:' + pct(v) +
            '"></div></div><span class="v">' + v.toFixed(2) + '</span></div>';
        }).join('') + '</div>';
      }).join('') + '</div>';
    $('detail').innerHTML = html;
  }

  function drawInternationalGuide() {
    $('guide-title').textContent = 'Band guide';
    const c = international(state.sel);
    const bands = state.intl ? state.intl.bands : [];
    $('guide').innerHTML = bands.map((b) =>
      '<div class="guide-row' + (c && c.band === b.band ? ' here' : '') + '"><div class="num" style="background:' +
      BAND_COLOURS[b.band].bg + ';color:' + BAND_COLOURS[b.band].fg + '">' + b.band + '</div>' +
      '<div><div class="title"><strong>' + esc(b.label) + '</strong>' +
      (c && c.band === b.band ? '<span>← ' + esc(names[state.sel] || state.sel) + '</span>' : '') + '</div>' +
      items(b.actions) + '</div></div>').join('');
  }

  // ---- Bilateral View panels ----
  function partnerRows(list) {
    return '<div class="partners">' + list.map(([key, p]) =>
      '<button type="button" class="prow" data-key="' + esc(key) + '"><span class="dot" style="background:' +
      RELATION_COLOURS[p[0]].bg + '"></span><span>' + esc(names[key] || key) + ' <span class="n">· ' +
      esc(relationBand(p[0]).label) + '</span></span><span class="n">' + p[2] + ' events</span></button>').join('') + '</div>';
  }

  function drawBilateralDetail() {
    const key = state.sel, name = names[key] || key;
    const all = Object.entries(partners());
    const judged = all.filter(([, p]) => p[0] > 0);
    const days = state.bil ? state.bil.window_days : 90;
    let html = '<div class="detail-head"><div><div class="iso">' + esc(key) + '</div><h2>' + esc(name) + '</h2></div>' +
      '<div class="badge" style="background:' + INK + ';color:#fff">Point of view</div></div>';
    if (!all.length) {
      html += '<p class="note">No recorded contact between ' + esc(name) + ' and other countries in the last ' + days + ' days.</p>';
      $('detail').innerHTML = html;
      return;
    }
    const worst = judged.slice().sort((a, b) => a[1][1] - b[1][1]).slice(0, 8);
    const best = judged.slice().sort((a, b) => b[1][1] - a[1][1]).slice(0, 8);
    html += '<p style="margin:0;color:#3d4654">Contact with ' + all.length + ' countries in the last ' + days +
      ' days; ' + judged.length + ' with enough contact to judge. Click a country (here or on the map) to see the map from its point of view.</p>' +
      '<div class="field"><span class="field-label">Most conflictual</span>' + partnerRows(worst) + '</div>' +
      '<div class="field"><span class="field-label">Most cooperative</span>' + partnerRows(best) + '</div>';
    $('detail').innerHTML = html;
    for (const b of $('detail').querySelectorAll('.prow')) b.addEventListener('click', () => select(b.dataset.key));
  }

  function drawBilateralGuide() {
    $('guide-title').textContent = 'Relation guide';
    const bands = state.bil ? state.bil.bands : [];
    $('guide').innerHTML = bands.map((b) =>
      '<div class="guide-row"><div class="num" style="background:' + (b.band ? RELATION_COLOURS[b.band].bg : HATCH_CSS) +
      ';color:' + (b.band ? RELATION_COLOURS[b.band].fg : INK) + '">' + (b.band || '–') + '</div>' +
      '<div><div class="title"><strong>' + esc(b.label) + '</strong></div><p>' + esc(b.description) + '</p></div></div>').join('') +
      '<p style="margin:8px 0 0;font-size:12px;color:#5b6472">Based on cooperative and conflictual events between the two countries, in either direction, over the last ' +
      (state.bil ? state.bil.window_days : 90) + ' days. A pair needs at least ' + (state.bil ? state.bil.min_events : 20) +
      ' events to be judged. Bands are cut at fixed percentiles of all pairs over the reference period.</p>';
  }

  // ---- Putting it together ----
  function render() {
    for (const b of document.querySelectorAll('[data-view]')) b.setAttribute('aria-pressed', String(b.dataset.view === state.view));
    const when = state.date ? longDate(state.date) : '';
    const name = names[state.sel] || state.sel;
    if (state.view === 'international') {
      $('map-title').textContent = 'Country bands' + (when ? ', ' + when : '');
      $('map-sub').textContent = 'Click a country for its detail.';
      drawInternationalDetail();
      drawInternationalGuide();
    } else {
      $('map-title').textContent = 'Relations of ' + name + (when ? ', ' + when : '');
      $('map-sub').textContent = 'Each country coloured by its recent contact with ' + name + '. Click any country to switch point of view.';
      drawBilateralDetail();
      drawBilateralGuide();
    }
    recolour();
    drawLegend();
    drawLabels();
  }

  function select(key) {
    for (const p of pathsByKey[state.sel] || []) p.classList.remove('sel');
    state.sel = key;
    for (const p of pathsByKey[key] || []) { p.classList.add('sel'); group.appendChild(p); }
    $('find-country').value = key;
    render();
  }

  async function loadDate(day) {
    state.date = day;
    try {
      const [intl, bil] = await Promise.all([
        getJSON(DATA + day + '/international.json'),
        getJSON(DATA + day + '/bilateral.json').catch(() => null)
      ]);
      state.intl = intl;
      state.bil = bil;
    } catch (error) {
      showError('Could not load the data for ' + day + ' (' + error.message + ').');
      state.intl = null;
      state.bil = null;
    }
    render();
  }

  function showError(text) {
    $('load-error').innerHTML = text + ' Export it with <code>python -m src.export_web</code>, serve the repository root with ' +
      '<code>python -m http.server 8000</code> and open <code>http://localhost:8000/src/web/</code>.';
    $('load-error').classList.remove('hidden');
  }

  for (const b of document.querySelectorAll('[data-view]')) {
    b.addEventListener('click', () => { state.view = b.dataset.view; render(); });
  }
  $('date-select').addEventListener('change', (e) => loadDate(e.target.value));

  async function start() {
    buildMap(await getJSON('world.json'));
    applyView();
    let index = null;
    try {
      index = await getJSON(DATA + 'index.json');
    } catch (error) {
      showError('No exported data found.');
    }
    const dates = index ? index.dates.slice().reverse() : [];
    $('date-select').innerHTML = dates.map((d, i) =>
      '<option value="' + d + '">' + (i === 0 ? 'Live: ' : 'Replay: ') + longDate(d) + '</option>').join('');
    if (dates.length) {
      await loadDate(dates[0]);
    } else {
      if (index) showError('No dates exported yet.');
      render();
    }
    select(state.sel);
  }

  start();
})();
