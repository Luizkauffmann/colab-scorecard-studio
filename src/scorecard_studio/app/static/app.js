// Scorecard Studio - interactive binning app.
// Ported from the Dataiku webapp (Luizkauffmann/scorecard-binning). All
// requests use relative URLs so they resolve through Colab's port proxy.
'use strict';

// ── API ──────────────────────────────────────────────────────────────
async function api(path, body) {
  const opts = { method: body ? 'POST' : 'GET', headers: { Accept: 'application/json' } };
  if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  const res = await fetch(path, opts);
  const ct = res.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    const t = await res.text();
    throw new Error('HTTP ' + res.status + ': ' + t.replace(/<[^>]+>/g, '').trim().slice(0, 160));
  }
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

// ── State ────────────────────────────────────────────────────────────
const S = {
  state: null,        // /api/state
  cur: null,          // payload of the open variable
  selected: new Set(),// selected bin groups
  chips: new Set(),   // selected categories
  assign: {},         // category -> group (working copy)
  nGroups: 0,
  showOthers: false,
  charts: { count: null, woe: null },
  modalCut: -1,
};

const COLORS = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad', '#16a085', '#d35400', '#2c3e50', '#c0392b',
                '#7f8c8d', '#1abc9c', '#9b59b6', '#f39c12', '#34495e', '#e84393', '#00b894'];
const CHIP_BG = ['#d6eaf8', '#d5f5e3', '#fdebd0', '#e8daef', '#d1f2eb', '#fae5d3', '#d6dbdf', '#fadbd8'];
const IV_CLASS = { 'Useless': 'ivu', 'Weak': 'ivw', 'Medium': 'ivm', 'Strong': 'ivs' };

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const pct = (x, d = 1) => (x == null ? '—' : (x * 100).toFixed(d) + '%');
const fmtNum = (x) => (x == null ? '—' : Math.abs(x) >= 1000 ? x.toLocaleString(undefined, { maximumFractionDigits: 2 }) : +x.toPrecision(6) + '');

// ── UI helpers ───────────────────────────────────────────────────────
function busy(on, msg) {
  $('busy').style.display = on ? 'flex' : 'none';
  if (msg) $('busyMsg').textContent = msg;
}
function toast(msg, type) {
  const el = $('toast');
  el.textContent = msg;
  el.className = type === 'error' ? 'error' : '';
  el.style.opacity = '1';
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.style.opacity = '0'; }, type === 'error' ? 7000 : 3200);
}
async function run(msg, fn) {
  busy(true, msg);
  try { return await fn(); }
  catch (e) { toast(e.message, 'error'); }
  finally { busy(false); }
}

// ── Init ─────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  wireStatic();
  try {
    S.state = await api('api/state');
    renderSidebar();
    const first = S.state.variables.find((v) => v.included) || S.state.variables[0];
    if (first) await openVar(first.name);
  } catch (e) { toast('Could not reach the app backend: ' + e.message, 'error'); }
});

function wireStatic() {
  $('binSlider').addEventListener('input', function () { $('binLbl').textContent = this.value; });
  $('btnFit').onclick = fitCurrent;
  $('btnReset').onclick = resetCurrent;
  $('btnOutput').onclick = createOutput;
  $('toggleOthers').onclick = () => { S.showOthers = !S.showOthers; renderSidebar(); };
  $('curIncluded').onchange = function () { setIncluded(S.cur.variable, this.checked); };
  $('tabStats').onclick = () => switchTab('stats');
  $('tabCat').onclick = () => switchTab('cat');
  $('btnAddGroup').onclick = () => { S.nGroups++; renderCatUI(); };
  $('btnApplyGroups').onclick = applyGrouping;
  $('modalCancel').onclick = closeModal;
  $('modalApply').onclick = applyThreshold;
  $('modalRemove').onclick = removeThreshold;
  $('thresholdModal').addEventListener('click', (e) => { if (e.target.id === 'thresholdModal') closeModal(); });
  $('modalThreshInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') applyThreshold(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
  window.addEventListener('resize', () => { if (S.cur && S.cur.dtype === 'numerical') renderCutLines(); });
}

// ── Sidebar ──────────────────────────────────────────────────────────
function renderSidebar() {
  const st = S.state;
  $('dsStats').innerHTML =
    '<span class="ds-stat ds-green">' + st.rows.toLocaleString() + ' Train rows</span>' +
    '<span class="ds-stat ds-blue">event rate ' + pct(st.event_rate) + '</span>';
  const ins = st.variables.filter((v) => v.included);
  const outs = st.variables.filter((v) => !v.included);
  $('nIncluded').textContent = ins.length;
  const maxIV = Math.max(0.01, ...st.variables.map((v) => v.iv || v.screen_iv || 0));
  $('varListIn').innerHTML = '';
  $('varListOut').innerHTML = '';
  ins.forEach((v) => $('varListIn').appendChild(varItem(v, maxIV)));
  outs.forEach((v) => $('varListOut').appendChild(varItem(v, maxIV)));
  if (!ins.length) $('varListIn').innerHTML = '<div class="sidebar-hint">No variables selected.</div>';
  $('othersWrap').style.display = S.showOthers ? '' : 'none';
  $('toggleOthers').textContent = (S.showOthers ? 'Hide' : 'Show') + ' other candidates (' + outs.length + ') ' + (S.showOthers ? '▴' : '▾');
  if (st.last_saved) { $('saveStatus').textContent = 'Saved to Drive at ' + st.last_saved; $('saveStatus').className = 'save-status ok'; }
  $('btnOutput').disabled = !ins.length;
  renderOutputInfo();
}

function varItem(v, maxIV) {
  const el = document.createElement('div');
  el.className = 'vi' + (S.cur && S.cur.variable === v.name ? ' active' : '');
  const iv = v.iv != null ? v.iv : v.screen_iv;
  const isNum = v.dtype === 'numerical';
  const statusPill = v.included ? (v.status === 'forced' ? '<span class="st-pill st-forced">forced</span>' : '')
    : '<span class="st-pill st-' + esc(v.status) + '">' + esc(v.status.replace('_', ' ')) + '</span>';
  el.innerHTML =
    '<input type="checkbox" ' + (v.included ? 'checked' : '') + (v.included || v.addable ? '' : ' disabled') +
    ' title="' + (v.addable || v.included ? 'Include in the output dataset' : esc(v.status) + ': decide in the config cell') + '">' +
    '<div class="vi-main"><div class="vi-name" title="' + esc(v.name) + '">' + esc(v.name) + '</div>' +
    '<div class="vi-meta"><span class="type-pill ' + (isNum ? 'tn' : 'tc') + '">' + (isNum ? 'N' : 'C') + '</span>' + statusPill +
    (v.edited ? '<span class="edited-dot" title="edited by hand"></span>' : '') +
    (v.n_warnings ? '<span class="warn-dot" title="' + v.n_warnings + ' warning(s)">⚠</span>' : '') + '</div>' +
    (iv != null ? '<div class="iv-bar-bg"><div class="iv-bar-fg" style="width:' + Math.min(100, iv / maxIV * 100).toFixed(0) + '%"></div></div>' : '') +
    '</div>' +
    '<span class="vi-iv">' + (iv != null ? iv.toFixed(3) : '') + '</span>';
  const cb = el.querySelector('input');
  cb.onclick = (e) => { e.stopPropagation(); setIncluded(v.name, cb.checked); };
  el.onclick = () => openVar(v.name);
  return el;
}

function renderOutputInfo() {
  const o = S.state.last_output;
  const el = $('outputInfo');
  if (!o) { el.style.display = 'none'; return; }
  el.style.display = '';
  const samples = Object.entries(o.samples).map(([k, n]) => k + ' ' + n.toLocaleString()).join(' · ');
  el.innerHTML = '<b>Output written at ' + esc(o.at) + '</b><br>' + o.rows.toLocaleString() + ' rows × ' +
    o.columns + ' columns (' + esc(samples) + ')<br>' + esc(o.paths.dataset);
}

async function setIncluded(name, included) {
  await run('Saving…', async () => {
    const r = await api('api/include', { name, included });
    S.state = r.state;
    if (S.cur && S.cur.variable === name) S.cur.included = included;
    renderSidebar();
    renderHeader();
  });
  if (S.state) renderSidebar();
}

// ── Open / refit ─────────────────────────────────────────────────────
async function openVar(name) {
  await run('Loading ' + name + '…', async () => {
    const r = await api('api/variable?name=' + encodeURIComponent(name));
    S.state = r.state;
    applyPayload(r.variable, true);
  });
}

function settingsFromUI() {
  const p = {
    max_bins: parseInt($('binSlider').value, 10),
    monotonic: $('monoSel').value,
    min_bin_size: (parseFloat($('minSize').value) || 0) / 100,
    min_bin_n_event: $('minEvents').value === '' ? null : parseInt($('minEvents').value, 10),
  };
  if (S.cur && S.cur.dtype === 'categorical') {
    p.cat_cutoff = $('catCut').value === '' ? null : parseFloat($('catCut').value) / 100;
  }
  return p;
}

function fitCurrent() {
  if (!S.cur) return;
  run('Running optimal binning…', async () => {
    const r = await api('api/fit', Object.assign({ name: S.cur.variable }, settingsFromUI()));
    afterChange(r, 'Re-binned with the new settings.');
  });
}

function resetCurrent() {
  if (!S.cur) return;
  run('Resetting…', async () => {
    const r = await api('api/reset', { name: S.cur.variable });
    afterChange(r, 'Back to the automatic binning.');
  });
}

function afterChange(r, msg) {
  S.state = r.state;
  applyPayload(r.variable, false);
  if (msg) toast(msg);
}

function applyPayload(v, resetSettings) {
  S.cur = v;
  S.selected.clear();
  S.chips.clear();
  $('emptyState').style.display = 'none';
  $('varContent').style.display = '';
  if (resetSettings) loadSettings(v);
  renderHeader();
  renderMetrics();
  renderTable();
  renderCharts();
  const isCat = v.dtype === 'categorical';
  $('tabCat').style.display = isCat ? '' : 'none';
  $('dragHint').style.display = isCat ? 'none' : '';
  $('leftChartTitle').textContent = isCat ? 'Rows per group (events darker)' : 'Distribution and cutoffs (Train)';
  if (isCat) buildCatUI(); else if ($('tabCat').classList.contains('active')) switchTab('stats');
  renderSidebar();
}

function loadSettings(v) {
  const s = Object.assign({}, S.state.fit_defaults, v.settings || {});
  $('binSlider').value = s.max_bins || 6; $('binLbl').textContent = $('binSlider').value;
  $('monoSel').value = s.monotonic || 'auto';
  $('minSize').value = s.min_bin_size != null ? +(s.min_bin_size * 100).toFixed(2) : '';
  $('minEvents').value = s.min_bin_n_event != null ? s.min_bin_n_event : '';
  $('catCut').value = s.cat_cutoff != null ? +(s.cat_cutoff * 100).toFixed(2) : '';
  const isNum = v.dtype === 'numerical';
  $('monoWrap').style.display = isNum ? '' : 'none';
  $('catCutWrap').style.display = isNum ? 'none' : '';
}

// ── Header + metrics ─────────────────────────────────────────────────
function renderHeader() {
  const v = S.cur; if (!v) return;
  $('varName').textContent = v.variable;
  $('varName').title = v.variable;
  const isNum = v.dtype === 'numerical';
  $('varPills').innerHTML =
    '<span class="pill ' + (isNum ? 'pn' : 'pc') + '">' + (isNum ? 'Numerical' : 'Categorical') + '</span>' +
    (v.edited ? '<span class="pill p-edited">edited</span>' : '<span class="pill p-auto">automatic</span>') +
    '<span class="pill p-auto">' + esc(v.status) + '</span>';
  $('curIncluded').checked = !!v.included;
  $('curIncluded').disabled = !v.included && !v.addable;
}

function renderMetrics() {
  const v = S.cur;
  $('mIV').textContent = v.iv.toFixed(3);
  $('mGini').textContent = pct(v.gini);
  $('mKS').textContent = pct(v.ks);
  $('mBins').textContent = v.bins.filter((b) => b.kind === 'regular').length;
  $('mMono').textContent = v.dtype === 'categorical' ? 'n/a'
    : v.is_monotonic ? (v.monotonic_direction === 'decreasing' ? '↓ Yes' : v.monotonic_direction === 'flat' ? '→ flat' : '↑ Yes') : '✗ No';
  const band = v.iv_band || '';
  $('ivInterp').textContent = band;
  $('ivInterp').className = 'iv-interp ' + (IV_CLASS[band] || (band.startsWith('Very') ? 'ivvs' : 'ivw'));
  $('screenIV').textContent = v.screen_iv != null ? 'IV at screening: ' + v.screen_iv.toFixed(3) : '';
  const flags = $('flags');
  flags.innerHTML = '';
  (v.flags || []).forEach((f) => {
    const li = document.createElement('li');
    li.className = f.level;
    li.textContent = f.text;
    flags.appendChild(li);
  });
}

// ── Bins table ───────────────────────────────────────────────────────
function regularBins() { return S.cur.bins.filter((b) => b.kind === 'regular'); }

function renderTable() {
  const v = S.cur;
  const tbody = $('binTbody');
  tbody.innerHTML = '';
  const reg = regularBins();
  const flagged = new Set((v.flags || []).filter((f) => f.level === 'error' && f.group != null).map((f) => f.group));
  v.bins.forEach((b) => {
    const fixed = b.kind !== 'regular';
    const ri = reg.indexOf(b);
    let trend = '';
    if (!fixed && ri > 0) {
      const d = b.woe - reg[ri - 1].woe;
      trend = d > 0.01 ? '<span class="woe-pos">↑</span>' : d < -0.01 ? '<span class="woe-neg">↓</span>' : '<span class="woe-zero">→</span>';
    }
    const wc = b.woe > 0.01 ? 'woe-pos' : b.woe < -0.01 ? 'woe-neg' : 'woe-zero';
    const rangeTxt = b.categories ? b.categories.map((c) => '<span class="cat-chip">' + esc(c) + '</span>').join('') : esc(b.label);
    let actions = '';
    if (!fixed && v.dtype === 'numerical') {
      if (ri < reg.length - 1) actions += '<button class="row-btn" data-act="edit" data-i="' + ri + '" title="Edit the upper boundary">✎</button> ';
      actions += '<button class="row-btn" data-act="split" data-g="' + b.group + '" title="Split at the bin median">⊕ split</button>';
    }
    const sel = S.selected.has(b.group);
    const tr = document.createElement('tr');
    tr.className = 'bin-row' + (sel ? ' selected' : '') + (fixed ? ' fixed' : '') + (flagged.has(b.group) ? ' flag-error' : '');
    tr.dataset.group = b.group;
    tr.innerHTML =
      '<td>' + (fixed ? '' : '<input type="checkbox"' + (sel ? ' checked' : '') + ' aria-label="select bin ' + b.group + '">') + '</td>' +
      '<td>' + (fixed ? esc(b.label) + '<span class="fixed-pill">fixed</span>' : 'G' + b.group) + '</td>' +
      '<td style="max-width:300px;white-space:normal">' + (fixed ? '' : rangeTxt) + '</td>' +
      '<td class="r">' + b.count.toLocaleString() + '</td>' +
      '<td class="r">' + pct(b.share) + '</td>' +
      '<td class="r">' + b.event_count.toLocaleString() + '</td>' +
      '<td class="r">' + pct(b.event_rate) + '</td>' +
      '<td class="r ' + wc + '">' + (b.woe == null ? '—' : b.woe.toFixed(4)) + '</td>' +
      '<td class="r">' + (b.iv_contribution == null ? '—' : b.iv_contribution.toFixed(4)) + '</td>' +
      '<td>' + trend + '</td>' +
      '<td style="white-space:nowrap">' + actions + '</td>';
    if (!fixed) {
      tr.addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON') return;
        const multi = e.shiftKey || e.ctrlKey || e.metaKey || e.target.tagName === 'INPUT';
        toggleRow(b.group, multi);
      });
    }
    tr.querySelectorAll('button[data-act]').forEach((btn) => {
      btn.onclick = (e) => {
        e.stopPropagation();
        if (btn.dataset.act === 'edit') openThreshModal(parseInt(btn.dataset.i, 10));
        else splitBin(parseInt(btn.dataset.g, 10));
      };
    });
    tbody.appendChild(tr);
  });
  updateBinToolbar();
}

function toggleRow(group, multi) {
  if (multi) { if (S.selected.has(group)) S.selected.delete(group); else S.selected.add(group); }
  else if (S.selected.size === 1 && S.selected.has(group)) S.selected.clear();
  else { S.selected.clear(); S.selected.add(group); }
  document.querySelectorAll('.bin-row').forEach((tr) => {
    const g = parseInt(tr.dataset.group, 10);
    const sel = S.selected.has(g);
    tr.classList.toggle('selected', sel);
    const cb = tr.querySelector('input[type=checkbox]');
    if (cb) cb.checked = sel;
  });
  updateBinToolbar();
  highlightBins();
}

function updateBinToolbar() {
  const tb = $('binToolbar');
  tb.querySelectorAll('.tb-action').forEach((el) => el.remove());
  const hint = $('binHint');
  const isNum = S.cur.dtype === 'numerical';
  if (!S.selected.size) {
    hint.style.display = '';
    hint.textContent = 'Click a bin to select it · Shift/Ctrl/⌘-click to select several · Missing and Special are fixed bins';
    return;
  }
  hint.style.display = 'none';
  const gs = Array.from(S.selected).sort((a, b) => a - b);
  const add = (el) => { el.classList.add('tb-action'); tb.appendChild(el); };
  const lbl = document.createElement('span');
  lbl.style.cssText = 'font-size:11px;color:#1a6fc4;font-weight:600';
  lbl.textContent = gs.length + ' selected (' + gs.map((g) => 'G' + g).join(', ') + ')';
  add(lbl);
  if (gs.length >= 2) {
    const adjacent = gs.every((g, i) => i === 0 || g === gs[i - 1] + 1);
    const btn = document.createElement('button');
    btn.className = 'primary';
    btn.textContent = 'Merge ' + gs.length + ' bins';
    btn.disabled = isNum && !adjacent;
    btn.title = btn.disabled ? 'Only adjacent numerical bins can be merged' : 'Merge into one bin';
    btn.onclick = () => mergeBins(gs);
    add(btn);
    if (btn.disabled) { const w = document.createElement('span'); w.style.cssText = 'font-size:10.5px;color:#c0392b'; w.textContent = 'select adjacent bins'; add(w); }
  }
  if (gs.length === 1 && isNum) {
    const reg = regularBins();
    const i = reg.findIndex((b) => b.group === gs[0]);
    if (i < reg.length - 1) { const b = document.createElement('button'); b.className = 'ghost'; b.textContent = '✎ Edit upper boundary'; b.onclick = () => openThreshModal(i); add(b); }
    const sp = document.createElement('button'); sp.className = 'ghost'; sp.textContent = '⊕ Split at median'; sp.onclick = () => splitBin(gs[0]); add(sp);
  }
  if (gs.length === 1 && !isNum) {
    const b = document.createElement('button'); b.className = 'ghost'; b.textContent = 'Edit categories →'; b.onclick = () => switchTab('cat'); add(b);
  }
  const clr = document.createElement('button');
  clr.textContent = 'Clear';
  clr.onclick = () => { S.selected.clear(); renderTable(); highlightBins(); };
  add(clr);
}

// ── Bin operations ───────────────────────────────────────────────────
function postCutoffs(cuts, msg) {
  return run('Recomputing…', async () => {
    const r = await api('api/cutoffs', { name: S.cur.variable, cutoffs: cuts });
    afterChange(r, msg);
  });
}
function mergeBins(groups) {
  run('Merging…', async () => {
    const r = await api('api/merge', { name: S.cur.variable, groups });
    afterChange(r, 'Merged ' + groups.length + ' bins.');
  });
}
function splitBin(group) {
  run('Splitting…', async () => {
    const r = await api('api/split', { name: S.cur.variable, group });
    afterChange(r, 'Split G' + group + ' at its median.');
  });
}

// ── Boundary modal ───────────────────────────────────────────────────
function cutBounds(i) {
  const c = S.cur.cutoffs; const h = S.cur.hist || {};
  const res = h.resolution || 0;
  const lo = i > 0 ? c[i - 1] : (h.min != null ? h.min - res : -Infinity);
  const hi = i < c.length - 1 ? c[i + 1] : (h.max != null ? h.max : Infinity);
  return [lo, hi];
}
function openThreshModal(i) {
  S.modalCut = i;
  const [lo, hi] = cutBounds(i);
  $('modalGrpA').textContent = 'G' + (i + 1); $('modalGrpA2').textContent = 'G' + (i + 1);
  $('modalGrpB').textContent = 'G' + (i + 2);
  $('modalCurrentVal').textContent = fmtNum(S.cur.cutoffs[i]);
  $('modalRange').textContent = '> ' + fmtNum(lo) + ' and < ' + fmtNum(hi);
  $('modalThreshInput').value = S.cur.cutoffs[i];
  $('thresholdModal').style.display = 'flex';
  setTimeout(() => $('modalThreshInput').select(), 30);
}
function closeModal() { $('thresholdModal').style.display = 'none'; }
function applyThreshold() {
  const val = parseFloat($('modalThreshInput').value);
  if (isNaN(val)) { toast('Enter a number.', 'error'); return; }
  const i = S.modalCut; const c = S.cur.cutoffs;
  const lo = i > 0 ? c[i - 1] : -Infinity;
  const hi = i < c.length - 1 ? c[i + 1] : Infinity;
  if (!(val > lo && val < hi)) { toast('The boundary must stay between its neighbours (' + fmtNum(lo) + ', ' + fmtNum(hi) + ').', 'error'); return; }
  const cuts = c.slice(); cuts[i] = val;
  closeModal();
  postCutoffs(cuts, 'Boundary set to ' + fmtNum(val) + '.');
}
function removeThreshold() {
  const i = S.modalCut;
  const cuts = S.cur.cutoffs.filter((_, k) => k !== i);
  closeModal();
  postCutoffs(cuts, 'Boundary removed: G' + (i + 1) + ' and G' + (i + 2) + ' merged.');
}

// ── Charts ───────────────────────────────────────────────────────────
function binIndexFor(value) {           // (lower, upper]: value <= cut goes left
  const c = S.cur.cutoffs; let i = 0;
  while (i < c.length && value > c[i]) i++;
  return i;
}
function destroyCharts() {
  ['count', 'woe'].forEach((k) => { if (S.charts[k]) { S.charts[k].destroy(); S.charts[k] = null; } });
}
function renderCharts() {
  destroyCharts();
  if (typeof Chart === 'undefined') { $('histNote').textContent = 'Charts need internet access to load Chart.js from jsdelivr.'; return; }
  if (S.cur.dtype === 'numerical') renderHistogram(); else renderGroupBars();
  renderWoeChart();
}

function renderHistogram() {
  const h = S.cur.hist;
  const n = h.counts.length;
  const centers = []; for (let i = 0; i < n; i++) centers.push((h.edges[i] + h.edges[i + 1]) / 2);
  const colorAt = (x) => COLORS[binIndexFor(x) % COLORS.length];
  const ev = centers.map((x, i) => ({ x, y: h.events[i] }));
  const ne = centers.map((x, i) => ({ x, y: h.counts[i] - h.events[i] }));
  S.charts.count = new Chart($('countChart'), {
    type: 'bar',
    data: { datasets: [
      { label: 'Events', data: ev, backgroundColor: centers.map((x) => colorAt(x)), stack: 's', barPercentage: 1, categoryPercentage: 1, grouped: false },
      { label: 'Non-events', data: ne, backgroundColor: centers.map((x) => colorAt(x) + '55'), stack: 's', barPercentage: 1, categoryPercentage: 1, grouped: false },
    ] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: {
        title: (items) => { const i = items[0].dataIndex; return h.integer ? 'value ' + fmtNum(centers[i]) : '(' + fmtNum(h.edges[i]) + ', ' + fmtNum(h.edges[i + 1]) + ']'; },
        label: (it) => it.dataset.label + ': ' + it.parsed.y.toLocaleString(),
        footer: (items) => { const i = items[0].dataIndex; return 'event rate ' + (h.counts[i] ? pct(h.events[i] / h.counts[i]) : '—') + ' · bin G' + (binIndexFor(centers[i]) + 1); },
      } } },
      scales: {
        x: { type: 'linear', min: h.edges[0], max: h.edges[n], stacked: true, ticks: { font: { size: 10 }, maxTicksLimit: 8, callback: (v) => fmtNum(v) } },
        y: { stacked: true, ticks: { font: { size: 10 } }, title: { display: true, text: 'rows', font: { size: 10 } } },
      },
      onResize: () => setTimeout(renderCutLines, 0),
    },
  });
  const notes = [];
  if (h.below || h.above) notes.push((h.below + h.above).toLocaleString() + ' extreme values outside the plotted range (1st–99th percentile) are still binned; drag or type cutoffs there with ✎.');
  notes.push(h.integer ? 'Integer values: a cutoff c puts values ≤ c in the lower bin.' : 'Cutoffs snap to ' + fmtNum(h.resolution) + '.');
  $('histNote').textContent = notes.join(' ');
  $('countChart').ondblclick = addCutAtEvent;
  requestAnimationFrame(renderCutLines);
}

function renderGroupBars() {
  const reg = regularBins();
  const labels = reg.map((b) => 'G' + b.group);
  const bg = reg.map((_, i) => COLORS[i % COLORS.length]);
  S.charts.count = new Chart($('countChart'), {
    type: 'bar',
    data: { labels, datasets: [
      { label: 'Events', data: reg.map((b) => b.event_count), backgroundColor: bg, stack: 's' },
      { label: 'Non-events', data: reg.map((b) => b.non_event_count), backgroundColor: bg.map((c) => c + '55'), stack: 's' },
    ] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { afterTitle: (items) => reg[items[0].dataIndex].label } } },
      onClick: (_, els) => { if (els.length) toggleRow(reg[els[0].index].group, false); },
      scales: { x: { stacked: true, ticks: { font: { size: 10 } } }, y: { stacked: true, ticks: { font: { size: 10 } } } },
    },
  });
  $('cutLines').innerHTML = '';
  $('histNote').textContent = '';
  $('countChart').ondblclick = null;
}

function renderWoeChart() {
  const bins = S.cur.bins.filter((b) => b.kind === 'regular' || b.count > 0);
  const labels = bins.map((b) => (b.kind === 'regular' ? 'G' + b.group : b.label));
  const woe = bins.map((b) => (b.woe == null ? 0 : +b.woe.toFixed(4)));
  const fill = bins.map((b, i) => (b.kind !== 'regular' ? '#b0b0b0' : woe[i] >= 0 ? '#e74c3c99' : '#2980b999'));
  S.charts.woe = new Chart($('woeChart'), {
    data: { labels, datasets: [
      { type: 'bar', label: 'WOE', data: woe, backgroundColor: fill, yAxisID: 'y', order: 2 },
      { type: 'line', label: 'Event rate', data: bins.map((b) => b.event_rate), borderColor: '#1a1a2e', backgroundColor: '#1a1a2e', pointRadius: 3, borderWidth: 1.5, yAxisID: 'y2', order: 1 },
    ] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: {
        afterTitle: (items) => bins[items[0].dataIndex].label,
        label: (it) => (it.dataset.label === 'WOE' ? 'WOE ' + it.parsed.y.toFixed(3) : 'event rate ' + pct(it.parsed.y)),
      } } },
      onClick: (_, els) => { if (els.length && bins[els[0].index].kind === 'regular') toggleRow(bins[els[0].index].group, false); },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: { ticks: { font: { size: 10 } }, title: { display: true, text: 'WOE (+ = riskier)', font: { size: 10 } } },
        y2: { position: 'right', min: 0, grid: { drawOnChartArea: false }, ticks: { font: { size: 10 }, callback: (v) => pct(v, 0) } },
      },
    },
  });
  highlightBins();
}

function highlightBins() {
  const sel = S.selected;
  const ch = S.charts.woe;
  if (!ch) return;
  const bins = S.cur.bins.filter((b) => b.kind === 'regular' || b.count > 0);
  ch.data.datasets[0].borderColor = bins.map((b) => (sel.has(b.group) ? '#1a6fc4' : 'transparent'));
  ch.data.datasets[0].borderWidth = bins.map((b) => (sel.has(b.group) ? 2.5 : 0));
  ch.update('none');
}

// ── Draggable cutoffs (numerical) ────────────────────────────────────
function xScale() { return S.charts.count && S.charts.count.scales.x; }
function cutToPixel(c) {
  // Integer data: draw between the bars of c and c+1.
  const h = S.cur.hist; const xs = xScale();
  const v = h.integer ? c + 0.5 : c;
  const clamped = Math.max(h.edges[0], Math.min(h.edges[h.edges.length - 1], v));
  return { px: xs.getPixelForValue(clamped), off: clamped !== v };
}
function pixelToCut(px) {
  const h = S.cur.hist; const xs = xScale();
  let v = xs.getValueForPixel(px);
  if (h.integer) return Math.round(v - 0.5);
  const r = h.resolution || 1;
  return +(Math.round(v / r) * r).toPrecision(12);
}
function renderCutLines() {
  const box = $('cutLines');
  box.innerHTML = '';
  if (!S.cur || S.cur.dtype !== 'numerical' || !S.charts.count) return;
  const chart = S.charts.count; const area = chart.chartArea; if (!area) return;
  const canvas = $('countChart');
  const offX = canvas.offsetLeft; const offY = canvas.offsetTop;
  const lastPx = [];                    // stagger labels of nearby cutoffs on up to 3 rows
  S.cur.cutoffs.forEach((cut, i) => {
    const pos = cutToPixel(cut);
    let level = 0;
    while (level < 2 && lastPx[level] != null && pos.px - lastPx[level] < 62) level++;
    lastPx[level] = pos.px;
    const line = document.createElement('div');
    line.className = 'cutoff-line' + (pos.off ? ' off-range' : '');
    line.style.left = (offX + pos.px) + 'px';
    line.style.top = (offY + area.top) + 'px';
    line.style.height = (area.bottom - area.top) + 'px';
    line.title = 'Cutoff ' + fmtNum(cut) + (pos.off ? ' (outside the plotted range)' : '') + ': drag, or click the label to type a value';
    const handle = document.createElement('div');
    handle.className = 'handle';
    handle.textContent = fmtNum(cut);
    handle.style.top = (level * 17 - 7) + 'px';
    line.appendChild(handle);
    line.addEventListener('mousedown', (e) => startDrag(e, i, line, handle));
    box.appendChild(line);
  });
}
function startDrag(e, i, line, handle) {
  e.preventDefault();
  const canvas = $('countChart');
  const rect = canvas.getBoundingClientRect();
  const area = S.charts.count.chartArea;
  const cuts = S.cur.cutoffs.slice();
  const h = S.cur.hist; const step = h.integer ? 1 : (h.resolution || 0);
  // Integer data: stop at the last whole number before a neighbouring cutoff.
  let lo = i > 0 ? cuts[i - 1] + step : -Infinity;
  let hi = i < cuts.length - 1 ? cuts[i + 1] - step : Infinity;
  if (h.integer) { lo = Math.ceil(lo); hi = Math.floor(hi); }
  const tip = $('cutTip');
  let moved = false; let value = cuts[i];
  line.classList.add('dragging');
  function onMove(ev) {
    const px = Math.max(area.left, Math.min(area.right, ev.clientX - rect.left));
    let v = pixelToCut(px);
    v = Math.max(lo, Math.min(hi, v));
    if (v !== value) moved = true;
    value = v;
    const p = cutToPixel(v);
    line.style.left = (canvas.offsetLeft + p.px) + 'px';
    handle.textContent = fmtNum(v);
    tip.style.display = 'block';
    tip.style.left = line.style.left;
    tip.style.top = (canvas.offsetTop + area.bottom + 4) + 'px';
    tip.textContent = (h.integer ? '≤ ' : '≤ ') + fmtNum(v);
  }
  function onUp() {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    tip.style.display = 'none';
    line.classList.remove('dragging');
    if (!moved) { openThreshModal(i); return; }
    cuts[i] = value;
    postCutoffs(cuts, 'Cutoff moved to ' + fmtNum(value) + '.');
  }
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}
function addCutAtEvent(e) {
  const canvas = $('countChart');
  const rect = canvas.getBoundingClientRect();
  const area = S.charts.count.chartArea;
  const px = e.clientX - rect.left;
  if (px < area.left || px > area.right) return;
  const v = pixelToCut(px);
  if (S.cur.cutoffs.some((c) => Math.abs(c - v) < 1e-12)) { toast('There is already a cutoff at ' + fmtNum(v) + '.'); return; }
  postCutoffs(S.cur.cutoffs.concat([v]).sort((a, b) => a - b), 'Cutoff added at ' + fmtNum(v) + '.');
}

// ── Categorical grouping ─────────────────────────────────────────────
function buildCatUI() {
  S.assign = {}; S.nGroups = 0; S.chips.clear();
  (S.cur.categories || []).forEach((c) => {
    if (c.group != null) { S.assign[c.value] = c.group; S.nGroups = Math.max(S.nGroups, c.group); }
  });
  renderCatUI();
}
function renderCatUI() {
  const box = $('catGroupsUI');
  box.innerHTML = '';
  const cats = S.cur.categories || [];
  for (let g = 1; g <= S.nGroups; g++) {
    const members = cats.filter((c) => S.assign[c.value] === g);
    const n = members.reduce((s, c) => s + c.count, 0);
    const ev = members.reduce((s, c) => s + c.events, 0);
    const row = document.createElement('div');
    row.className = 'cat-group-row';
    row.innerHTML = '<div class="cat-group-lbl" style="color:' + COLORS[(g - 1) % COLORS.length] + '">Group ' + g +
      '<small>' + (n ? n.toLocaleString() + ' rows · ' + pct(ev / n) : 'empty') + '</small></div>' +
      '<div class="chips-zone"></div>';
    const zone = row.querySelector('.chips-zone');
    members.forEach((c) => zone.appendChild(chip(c, CHIP_BG[(g - 1) % CHIP_BG.length])));
    const mv = document.createElement('button');
    mv.textContent = '← Move here';
    mv.style.cssText = 'font-size:11px;flex-shrink:0';
    mv.onclick = () => moveChips(g);
    row.appendChild(mv);
    box.appendChild(row);
  }
}
function chip(c, bg) {
  const el = document.createElement('span');
  el.className = 'chip' + (S.chips.has(c.value) ? ' selected' : '');
  el.style.background = bg;
  el.title = c.count.toLocaleString() + ' rows, ' + c.events.toLocaleString() + ' events';
  el.innerHTML = esc(c.value) + '<span class="chip-rate">' + pct(c.event_rate) + ' · ' + c.count.toLocaleString() + '</span>';
  el.onclick = (e) => {
    if (e.ctrlKey || e.metaKey || e.shiftKey) { if (S.chips.has(c.value)) S.chips.delete(c.value); else S.chips.add(c.value); }
    else if (S.chips.has(c.value) && S.chips.size === 1) S.chips.clear();
    else { S.chips.clear(); S.chips.add(c.value); }
    renderCatUI();
  };
  return el;
}
function moveChips(g) {
  if (!S.chips.size) { toast('Select one or more categories first.'); return; }
  S.chips.forEach((v) => { S.assign[v] = g; });
  S.chips.clear();
  renderCatUI();
}
function applyGrouping() {
  // Renumber groups 1..k in on-screen order, dropping empty ones.
  const used = Array.from(new Set(Object.values(S.assign))).sort((a, b) => a - b);
  const remap = {}; used.forEach((g, i) => { remap[g] = i + 1; });
  const assignments = {};
  Object.entries(S.assign).forEach(([c, g]) => { assignments[c] = remap[g]; });
  run('Applying grouping…', async () => {
    const r = await api('api/categories', { name: S.cur.variable, assignments });
    afterChange(r, 'Grouping applied.');
    switchTab('cat');
  });
}

function switchTab(tab) {
  $('tabStats').classList.toggle('active', tab === 'stats');
  $('tabCat').classList.toggle('active', tab === 'cat');
  $('panelStats').style.display = tab === 'stats' ? '' : 'none';
  $('panelCat').style.display = tab === 'cat' ? '' : 'none';
  $('binToolbar').style.display = tab === 'stats' ? '' : 'none';
}

// ── Output dataset ───────────────────────────────────────────────────
function createOutput() {
  const n = S.state.included.length;
  run('Writing the output dataset…', async () => {
    const r = await api('api/output', {});
    S.state = r.state;
    renderSidebar();
    toast('Output dataset written: ' + r.output.rows.toLocaleString() + ' rows, ' + n + ' variables as original + opt_ + woe_.');
  });
}
