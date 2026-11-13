// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The WM^MLLM Authors
import { ViewerError, requestedWorld, validateWorld, PRESENTATION_LABELS } from './world.js';
import { PanoramaRenderer } from './renderer.js';

const byId = (id) => document.getElementById(id);
const app = byId('app');
const canvas = byId('panorama');
const video = byId('transition');
const stage = byId('stage');
const actionsElement = byId('actions');
const revisitsElement = byId('revisits');
const undoButton = byId('undo');
const skipPreference = byId('skip-clips');
const errorElement = byId('error');
const messageElement = byId('message');
const presentationElement = byId('presentation');
const playButton = byId('play-clip');
const skipButton = byId('skip-clip');
const playbackControls = byId('playback-controls');
const endpointPanel = byId('endpoint-panel');
const endpoint = byId('endpoint');

let world = null;
let baseUrl = null;
let renderer = null;
let nodeId = null;
let camera = { yaw: 0, pitch: 0, hfov: 90 };
let busy = true;
let unavailable = false;
let disposed = false;
let cancelPlayback = null;
const history = [];
const visited = new Map();
const pointers = new Map();
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const wrap = (yaw) => ((yaw + 180) % 360 + 360) % 360 - 180;
const snapshot = () => ({ nodeId, camera: { ...camera } });
const mediaUrl = (path) => new URL(path, baseUrl).href;

function reportError(error) {
  errorElement.textContent = error instanceof ViewerError ? error.message : 'The viewer could not complete this operation. The displayed state was retained.';
  errorElement.hidden = false;
}

function clearError() {
  errorElement.textContent = '';
  errorElement.hidden = true;
}

function updateDisabled() {
  for (const button of actionsElement.querySelectorAll('button')) button.disabled = busy || unavailable;
  for (const button of revisitsElement.querySelectorAll('button')) {
    button.disabled = busy || unavailable || button.dataset.nodeId === nodeId;
  }
  undoButton.disabled = busy || unavailable || history.length === 0;
  skipPreference.disabled = busy || unavailable;
}

function setBusy(value) {
  busy = value;
  app.dataset.busy = String(value);
  stage.setAttribute('aria-busy', String(value));
  if (value) {
    pointers.clear();
    canvas.classList.remove('dragging');
  }
  updateDisabled();
}

function updateCameraReadout() {
  canvas.dataset.yaw = String(camera.yaw);
  canvas.dataset.pitch = String(camera.pitch);
  canvas.dataset.hfov = String(camera.hfov);
  byId('camera-value').textContent = `Yaw ${camera.yaw.toFixed(1)}° · Pitch ${camera.pitch.toFixed(1)}° · FOV ${camera.hfov.toFixed(1)}°`;
}

function draw() {
  if (disposed || unavailable || !nodeId || !renderer) return false;
  const drawn = renderer.draw(camera, world.nodes.get(nodeId).viewport);
  updateCameraReadout();
  return drawn;
}

function changeCamera(view) {
  camera = {
    yaw: wrap(view.yaw),
    pitch: clamp(view.pitch, -90, 90),
    hfov: clamp(view.hfov, 1, 179),
  };
  if (nodeId) visited.set(nodeId, { ...camera });
  draw();
}

function updateEndpoint() {
  if (nodeId && endpointPanel.open) {
    const url = mediaUrl(world.nodes.get(nodeId).endpoint);
    if (endpoint.src !== url) endpoint.src = url;
  } else endpoint.removeAttribute('src');
}

function renderControls() {
  actionsElement.replaceChildren();
  const actions = world.outgoing.get(nodeId);
  for (const action of actions) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.actionId = action.id;
    const label = document.createElement('span');
    label.className = 'action-label';
    label.textContent = action.label;
    const detail = document.createElement('span');
    detail.className = 'action-detail';
    const operations = action.operations.length ? action.operations.join(', ').replaceAll('_', ' ') : 'no operation';
    const status = action.status === 'accepted' ? 'committed' : action.status;
    detail.textContent = `${operations} · ${status} · ${action.semantic} · ${PRESENTATION_LABELS[action.presentation.kind]}`;
    button.append(label, detail);
    button.addEventListener('click', () => { void runAction(action); });
    actionsElement.append(button);
  }
  if (!actions.length) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = 'No further action is prepared from this state.';
    actionsElement.append(empty);
  }

  revisitsElement.replaceChildren();
  for (const id of visited.keys()) {
    const node = world.nodes.get(id);
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.nodeId = id;
    button.textContent = `${id} · ${node.place_id} · revision ${node.revision}`;
    if (id === nodeId) button.setAttribute('aria-current', 'true');
    button.addEventListener('click', () => { void restoreVisited(id); });
    revisitsElement.append(button);
  }
  const node = world.nodes.get(nodeId);
  const seam = node.seam_status === 'seam_unresolved' ? 'unresolved' : node.seam_status.replaceAll('_', ' ');
  byId('node-summary').textContent = `${node.id} · Place ${node.place_id} · Revision ${node.revision} · Seam ${seam}`;
  updateEndpoint();
  updateDisabled();
}

function showNode(targetId, record, view) {
  // Only pose initialization needs normalization. Saved history cameras already
  // contain the exact finite values displayed before an action.
  if (![view.yaw, view.pitch, view.hfov].every(Number.isFinite)) {
    throw new ViewerError('The saved camera is invalid. Reload the viewer.');
  }
  if (!renderer.show(record, view, world.nodes.get(targetId).viewport)) {
    throw new ViewerError('The panorama could not be displayed. The previous state was retained.');
  }
  nodeId = targetId;
  camera = { ...view };
  visited.set(nodeId, { ...camera });
  app.dataset.nodeId = nodeId;
  updateCameraReadout();
  app.dataset.ready = 'true';
  renderControls();
}

function initialCamera(node) {
  return {
    yaw: wrap(node.pose.yaw),
    pitch: node.pose.pitch,
    hfov: clamp(node.pose.hfov, 1, 179),
  };
}

/** Play only the actual supplied clip. Errors preserve the current state. */
function playTransition(action) {
  return new Promise((resolve, reject) => {
    let finished = false;
    let loaded = false;
    let starting = false;
    let started = false;
    let timer = 0;
    let watchdog = 0;
    let lastTime = -1;
    let lastProgress = performance.now();

    const finish = (result) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      clearInterval(watchdog);
      video.removeEventListener('loadeddata', onData);
      video.removeEventListener('ended', onEnded);
      video.removeEventListener('error', onError);
      playButton.removeEventListener('click', begin);
      skipButton.removeEventListener('click', onSkip);
      cancelPlayback = null;
      video.pause();
      video.hidden = true;
      video.removeAttribute('src');
      video.load();
      playbackControls.hidden = true;
      playButton.hidden = true;
      if (result === 'error' || result === 'cancelled') {
        reject(new ViewerError('The prepared clip could not be played. The displayed state was retained. Select Skip prepared clips to view its recorded result.'));
      } else resolve(result);
    };
    const onEnded = () => finish('played');
    const onError = () => finish('error');
    const onSkip = () => finish('skipped');
    const begin = async () => {
      if (finished || starting) return;
      starting = true;
      try {
        // This call also runs directly in the Play button's real click gesture.
        await video.play();
        if (finished) return;
        started = true;
        playButton.hidden = true;
        lastProgress = performance.now();
        clearTimeout(timer);
        const limit = Math.min(300000, Math.max(15000, action.presentation.duration_seconds * 1000 + 12000));
        timer = setTimeout(onError, limit);
      } catch (error) {
        if (finished) return;
        if (error?.name === 'NotAllowedError') {
          playButton.hidden = false;
          messageElement.textContent = 'This browser requires a gesture. Choose Play clip or Skip to result.';
          playButton.focus({ preventScroll: true });
        } else finish('error');
      } finally { starting = false; }
    };
    const onData = () => {
      if (finished || loaded) return;
      loaded = true;
      clearTimeout(timer);
      video.hidden = false;
      void begin();
    };

    cancelPlayback = () => finish('cancelled');
    playbackControls.hidden = false;
    playButton.hidden = true;
    skipButton.addEventListener('click', onSkip);
    playButton.addEventListener('click', begin);
    video.addEventListener('loadeddata', onData);
    video.addEventListener('ended', onEnded);
    video.addEventListener('error', onError);
    video.muted = true;
    video.defaultMuted = true;
    timer = setTimeout(onError, 12000);
    watchdog = setInterval(() => {
      if (!started || finished) return;
      if (video.currentTime !== lastTime) {
        lastTime = video.currentTime;
        lastProgress = performance.now();
      } else if (performance.now() - lastProgress > 12000) finish('error');
    }, 500);
    messageElement.textContent = 'Loading the prepared clip…';
    video.src = mediaUrl(action.presentation.video);
    video.load();
  });
}

async function runAction(action) {
  if (busy || unavailable || action.from !== nodeId) return;
  clearError();
  const before = snapshot();
  const targetId = action.to ?? nodeId;
  const target = world.nodes.get(targetId);
  let record = null;
  setBusy(true);
  presentationElement.textContent = PRESENTATION_LABELS[action.presentation.kind];
  try {
    messageElement.textContent = 'Loading the recorded result…';
    record = await renderer.prepare(mediaUrl(target.panorama));
    if (disposed || unavailable) return;
    let result = 'none';
    if (action.presentation.kind !== 'none') {
      result = skipPreference.checked ? 'skipped' : await playTransition(action);
    }
    if (disposed || unavailable) return;
    if (action.status === 'accepted') {
      showNode(targetId, record, initialCamera(target));
      history.push(before);
      const skipped = result === 'skipped' ? 'Prepared clip skipped. ' : '';
      const uncertain = action.semantic === 'UNCERTAIN' ? ' Semantic check: UNCERTAIN.' : '';
      messageElement.textContent = `${skipped}${action.label} — showing ${targetId}.${uncertain}`;
      if (action.presentation.kind === 'none') messageElement.textContent += ` ${action.presentation.reason}`;
    } else {
      renderer.discard(record);
      const label = action.status === 'held' ? 'held' : 'failed';
      messageElement.textContent = `Recorded action ${label}. State ${nodeId} was retained. ${action.presentation.reason ?? ''}`.trim();
    }
  } catch (error) {
    renderer?.discard(record);
    if (!disposed && !unavailable) {
      reportError(error);
      messageElement.textContent = `State ${before.nodeId} was retained.`;
    }
  } finally {
    renderer?.discard(record);
    if (!disposed) setBusy(false);
  }
}

async function restore(target, isUndo) {
  if (busy || unavailable || !visited.has(target.nodeId)) return;
  clearError();
  const before = snapshot();
  let record = null;
  setBusy(true);
  try {
    record = await renderer.prepare(mediaUrl(world.nodes.get(target.nodeId).panorama));
    if (disposed || unavailable) return;
    showNode(target.nodeId, record, target.camera);
    if (isUndo) history.pop();
    else history.push(before);
    presentationElement.textContent = PRESENTATION_LABELS.none;
    messageElement.textContent = isUndo ? `Restored ${target.nodeId} and its previous camera.` : `Restored visited state ${target.nodeId}.`;
  } catch (error) {
    if (!disposed) reportError(error);
  } finally {
    renderer?.discard(record);
    if (!disposed) setBusy(false);
  }
}

async function restoreVisited(id) {
  if (!visited.has(id) || id === nodeId) return;
  await restore({ nodeId: id, camera: { ...visited.get(id) } }, false);
}

undoButton.addEventListener('click', () => {
  if (history.length) void restore(history.at(-1), true);
});
endpointPanel.addEventListener('toggle', updateEndpoint);
endpoint.addEventListener('error', () => {
  if (endpoint.hasAttribute('src')) reportError(new ViewerError('The stored endpoint image could not be loaded.'));
});

canvas.addEventListener('pointerdown', (event) => {
  if (busy || unavailable || (event.pointerType === 'mouse' && event.button !== 0)) return;
  event.preventDefault();
  canvas.focus({ preventScroll: true });
  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  canvas.setPointerCapture(event.pointerId);
  canvas.classList.add('dragging');
});
canvas.addEventListener('pointermove', (event) => {
  if (busy || unavailable || !pointers.has(event.pointerId)) return;
  const previous = pointers.get(event.pointerId);
  const next = { x: event.clientX, y: event.clientY };
  if (pointers.size === 2) {
    const other = [...pointers.entries()].find(([id]) => id !== event.pointerId)[1];
    const oldDistance = Math.hypot(previous.x - other.x, previous.y - other.y);
    const newDistance = Math.hypot(next.x - other.x, next.y - other.y);
    if (oldDistance > 1 && newDistance > 1) changeCamera({ ...camera, hfov: clamp(camera.hfov * oldDistance / newDistance, 30, 150) });
  } else if (pointers.size === 1) {
    const scale = camera.hfov / Math.max(1, canvas.clientWidth);
    changeCamera({ ...camera, yaw: camera.yaw - (next.x - previous.x) * scale, pitch: clamp(camera.pitch + (next.y - previous.y) * scale, -89.9, 89.9) });
  }
  pointers.set(event.pointerId, next);
});
const releasePointer = (event) => {
  pointers.delete(event.pointerId);
  if (!pointers.size) canvas.classList.remove('dragging');
};
canvas.addEventListener('pointerup', releasePointer);
canvas.addEventListener('pointercancel', releasePointer);
canvas.addEventListener('lostpointercapture', releasePointer);
canvas.addEventListener('wheel', (event) => {
  if (busy || unavailable) return;
  event.preventDefault();
  changeCamera({ ...camera, hfov: clamp(camera.hfov + event.deltaY * .04, 30, 150) });
}, { passive: false });
canvas.addEventListener('keydown', (event) => {
  if (busy || unavailable || event.ctrlKey || event.metaKey || event.altKey) return;
  const step = event.shiftKey ? 15 : 5;
  const view = { ...camera };
  switch (event.key.toLowerCase()) {
    case 'arrowleft': case 'a': view.yaw -= step; break;
    case 'arrowright': case 'd': view.yaw += step; break;
    case 'arrowup': case 'w': view.pitch = clamp(view.pitch + step, -89.9, 89.9); break;
    case 'arrowdown': case 's': view.pitch = clamp(view.pitch - step, -89.9, 89.9); break;
    case '+': case '=': view.hfov = clamp(view.hfov - 5, 30, 150); break;
    case '-': case '_': view.hfov = clamp(view.hfov + 5, 30, 150); break;
    case 'r': Object.assign(view, initialCamera(world.nodes.get(nodeId))); break;
    case 'z': if (history.length) void restore(history.at(-1), true); event.preventDefault(); return;
    default: return;
  }
  event.preventDefault();
  changeCamera(view);
});
app.addEventListener('keydown', (event) => {
  if (cancelPlayback && (event.key === 'Escape' || (event.code === 'Space' && event.target === canvas))) {
    event.preventDefault();
    skipButton.click();
  }
});
canvas.addEventListener('webglcontextlost', (event) => {
  event.preventDefault();
  unavailable = true;
  app.dataset.ready = 'false';
  cancelPlayback?.();
  reportError(new ViewerError('The WebGL2 context was lost. Reload the viewer to resume playback.'));
  setBusy(false);
});

const resizeObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(draw) : null;
resizeObserver?.observe(stage);
window.addEventListener('resize', draw);
window.addEventListener('pagehide', (event) => {
  if (event.persisted) return;
  disposed = true;
  cancelPlayback?.();
  resizeObserver?.disconnect();
  window.removeEventListener('resize', draw);
  renderer?.destroy();
});

async function start() {
  try {
    const id = requestedWorld(window.location.search, document.querySelector('meta[name="wmmllm-default-world"]')?.content);
    baseUrl = new URL(`../worlds/${id}/`, import.meta.url);
    const response = await fetch(new URL('world.json', baseUrl), { credentials: 'omit', redirect: 'error' });
    if (!response.ok) throw new ViewerError('The world manifest could not be loaded. Serve the viewer from its directory and check the selected world ID.');
    const source = await response.text();
    if (source.length > 8 * 1024 * 1024) throw new ViewerError('The world manifest exceeds the viewer size limit.');
    let manifest;
    try { manifest = JSON.parse(source); }
    catch { throw new ViewerError('The world manifest is not valid JSON. No media was loaded.'); }
    world = validateWorld(manifest, id);
    byId('world-title').textContent = manifest.title;
    document.title = `${manifest.title} · WM^MLLM viewer`;
    byId('fixture').hidden = !manifest.procedural_fixture;
    byId('attribution').textContent = manifest.attribution;
    byId('recorded-final').textContent = `Recorded final state: ${manifest.current_node_id}`;
    skipPreference.checked = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    byId('motion-note').hidden = !skipPreference.checked;
    renderer = new PanoramaRenderer(canvas);
    const initial = world.nodes.get(manifest.initial_node_id);
    const record = await renderer.prepare(mediaUrl(initial.panorama));
    if (disposed || unavailable) return;
    showNode(initial.id, record, initialCamera(initial));
    messageElement.textContent = 'Choose a prepared action or look around the panorama.';
    setBusy(false);
  } catch (error) {
    unavailable = true;
    reportError(error);
    messageElement.textContent = 'Playback is unavailable.';
    setBusy(false);
  }
}

void start();
