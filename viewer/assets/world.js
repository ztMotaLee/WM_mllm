// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The WM^MLLM Authors

/** A deliberately safe error message suitable for the public viewer. */
export class ViewerError extends Error {}

const ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/;
const SEGMENT = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;
const OPERATIONS = new Set(['rotate', 'revisit', 'navigate', 'local_edit', 'global_change', 'subject_action']);
const SEMANTICS = new Set(['PASS', 'FAIL', 'UNCERTAIN', 'NOT_CHECKED']);
const SEAMS = new Set(['resolved', 'seam_unresolved', 'not_checked']);
const fail = (field) => { throw new ViewerError(`Invalid world manifest: ${field}. No media was loaded.`); };
const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const integer = (value) => Number.isSafeInteger(value);
const identifier = (value) => typeof value === 'string' && ID.test(value);
const text = (value, limit) => typeof value === 'string' && value.length > 0 && value.length <= limit && !/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value);

function shape(value, keys, field) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail(field);
  const found = Object.keys(value);
  if (found.length !== keys.length || !keys.every((key) => Object.hasOwn(value, key))) fail(field);
}

/** Only literal, normalized, local POSIX media paths are accepted. */
export function safeMediaPath(path) {
  if (typeof path !== 'string' || path.length > 512) return false;
  const parts = path.split('/');
  return parts.length >= 2 && parts[0] === 'media' && parts.every((part) => SEGMENT.test(part));
}

export function requestedWorld(search, defaultWorld = 'procedural') {
  if (!search) return identifier(defaultWorld) ? defaultWorld : 'procedural';
  const match = /^\?world=([A-Za-z0-9][A-Za-z0-9_.-]{0,79})$/.exec(search);
  if (!match) throw new ViewerError('Invalid world selection. Use ?world= followed by a local world ID.');
  return match[1];
}

function validatePose(pose) {
  shape(pose, ['yaw', 'pitch', 'hfov'], 'node pose');
  if (!finite(pose.yaw) || !finite(pose.pitch) || !finite(pose.hfov) ||
      Math.abs(pose.pitch) > 90 || pose.hfov <= 0 || pose.hfov >= 180) fail('node pose');
}

function validateViewport(viewport) {
  shape(viewport, ['width', 'height', 'content_box', 'padding_ltrb'], 'node viewport');
  const { width, height, content_box: box, padding_ltrb: padding } = viewport;
  if (!integer(width) || !integer(height) || width < 1 || height < 1 || width > 32768 || height > 32768 ||
      !Array.isArray(box) || box.length !== 4 || !box.every(integer) ||
      !Array.isArray(padding) || padding.length !== 4 || !padding.every((n) => integer(n) && n >= 0)) fail('node viewport');
  if (box[0] < 0 || box[1] < 0 || box[2] > width || box[3] > height ||
      box[0] >= box[2] || box[1] >= box[3] ||
      padding[0] !== box[0] || padding[1] !== box[1] ||
      padding[2] !== width - box[2] || padding[3] !== height - box[3]) fail('node viewport bounds');
}

function validatePresentation(presentation) {
  shape(presentation, ['kind', 'video', 'reason', 'num_frames', 'fps', 'duration_seconds'], 'action presentation');
  const { kind, video, reason, num_frames: frames, fps, duration_seconds: duration } = presentation;
  if (!['rendered', 'generated', 'none'].includes(kind) ||
      !(reason === null || text(reason, 2000)) ||
      !(frames === null || (integer(frames) && frames > 0)) ||
      !(fps === null || (finite(fps) && fps > 0)) ||
      !(duration === null || (finite(duration) && duration >= 0))) fail('action presentation');
  if (kind === 'none') {
    if (video !== null || !text(reason, 2000)) fail('absent video');
  } else {
    if (!safeMediaPath(video) || !/\.mp4$/i.test(video) || frames === null || fps === null || duration === null ||
        Math.abs(duration - frames / fps) > 1e-6 * Math.max(1, duration)) fail('video path or timing');
  }
}

function freeze(value) {
  if (value && typeof value === 'object') {
    for (const item of Object.values(value)) freeze(item);
    Object.freeze(value);
  }
  return value;
}

/** Validate every field, path and graph edge before any image/video request. */
export function validateWorld(manifest, expectedId) {
  shape(manifest, ['schema', 'id', 'title', 'procedural_fixture', 'attribution', 'initial_node_id', 'current_node_id', 'nodes', 'actions'], 'top-level fields');
  if (manifest.schema !== 'wmmllm.world.v1' || !identifier(manifest.id) || manifest.id !== expectedId ||
      !text(manifest.title, 500) || typeof manifest.procedural_fixture !== 'boolean' || !text(manifest.attribution, 500) ||
      !identifier(manifest.initial_node_id) || !identifier(manifest.current_node_id) ||
      !Array.isArray(manifest.nodes) || manifest.nodes.length < 1 || manifest.nodes.length > 10000 ||
      !Array.isArray(manifest.actions) || manifest.actions.length > 50000) fail('world identity or graph');

  const nodes = new Map();
  for (const node of manifest.nodes) {
    shape(node, ['id', 'parent_id', 'place_id', 'revision', 'pose', 'viewport', 'panorama', 'endpoint', 'seam_status'], 'node fields');
    if (!identifier(node.id) || nodes.has(node.id) || !(node.parent_id === null || identifier(node.parent_id)) ||
        !identifier(node.place_id) || !integer(node.revision) || node.revision < 0 || !SEAMS.has(node.seam_status) ||
        !safeMediaPath(node.panorama) || !safeMediaPath(node.endpoint) ||
        !/\.(png|jpe?g|webp)$/i.test(node.panorama) || !/\.(png|jpe?g|webp)$/i.test(node.endpoint)) fail('node identity or media');
    validatePose(node.pose);
    validateViewport(node.viewport);
    nodes.set(node.id, node);
  }
  if (!nodes.has(manifest.initial_node_id) || !nodes.has(manifest.current_node_id) ||
      nodes.get(manifest.initial_node_id).parent_id !== null) fail('initial or current node');
  const checked = new Set([manifest.initial_node_id]);
  for (const node of nodes.values()) {
    const chain = new Set();
    let current = node;
    while (!checked.has(current.id)) {
      if (chain.has(current.id) || current.parent_id === null || !nodes.has(current.parent_id)) fail('node parent graph');
      chain.add(current.id);
      current = nodes.get(current.parent_id);
    }
    for (const id of chain) checked.add(id);
  }

  const actionIds = new Set();
  const outgoing = new Map(manifest.nodes.map((node) => [node.id, []]));
  for (const action of manifest.actions) {
    shape(action, ['id', 'from', 'to', 'label', 'operations', 'status', 'semantic', 'presentation'], 'action fields');
    if (!identifier(action.id) || actionIds.has(action.id) || !nodes.has(action.from) ||
        !text(action.label, 500) || !Array.isArray(action.operations) ||
        !action.operations.every((operation) => OPERATIONS.has(operation)) || !SEMANTICS.has(action.semantic) ||
        !['accepted', 'held', 'failed'].includes(action.status)) fail('action identity or status');
    if ((action.status === 'accepted' && (!nodes.has(action.to) || !['PASS', 'UNCERTAIN'].includes(action.semantic))) ||
        (action.status === 'held' && action.to !== action.from) ||
        (action.status === 'failed' && action.to !== null) ||
        (action.status !== 'accepted' && !['FAIL', 'NOT_CHECKED'].includes(action.semantic))) fail('action target or semantic status');
    validatePresentation(action.presentation);
    actionIds.add(action.id);
    outgoing.get(action.from).push(action);
  }
  const reachable = new Set([manifest.initial_node_id]);
  const queue = [manifest.initial_node_id];
  for (let index = 0; index < queue.length; index += 1) {
    for (const action of outgoing.get(queue[index])) {
      if (action.status === 'accepted' && !reachable.has(action.to)) {
        reachable.add(action.to);
        queue.push(action.to);
      }
    }
  }
  if (reachable.size !== nodes.size) fail('unreachable graph node');
  freeze(manifest);
  return { manifest, nodes, outgoing };
}

export const PRESENTATION_LABELS = Object.freeze({
  rendered: 'CPU panorama rendering',
  generated: 'Generated video',
  none: 'No generated video',
});
