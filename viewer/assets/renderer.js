// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The WM^MLLM Authors
import { ViewerError } from './world.js';

const DEG = Math.PI / 180;
const VERTEX = `#version 300 es
in vec2 aPosition;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }`;

// Rectilinear rays: pitch positive up, yaw positive right, ERP center at yaw 0.
const FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D uPanorama;
uniform vec2 uCenter;
uniform float uFocal;
uniform vec4 uRotation;
out vec4 outColor;
const float PI = 3.141592653589793;
void main() {
  vec3 d = normalize(vec3(gl_FragCoord.xy - uCenter, uFocal));
  float dy = d.y * uRotation.x + d.z * uRotation.y;
  float dz = -d.y * uRotation.y + d.z * uRotation.x;
  float X = d.x * uRotation.z + dz * uRotation.w;
  float Z = -d.x * uRotation.w + dz * uRotation.z;
  vec2 uv = vec2(atan(X, Z) / (2.0 * PI) + 0.5,
                 0.5 - asin(clamp(dy, -1.0, 1.0)) / PI);
  outColor = vec4(texture(uPanorama, uv).rgb, 1.0);
}`;

/** A single panorama texture. State changes are hard cuts. */
export class PanoramaRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.current = null;
    this.pending = new Set();
    this.disposed = false;
    const gl = canvas.getContext('webgl2', {
      alpha: false, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false, powerPreference: 'default',
    });
    if (!gl) throw new ViewerError('WebGL2 is unavailable. Open this viewer in a browser with WebGL2 enabled.');
    this.gl = gl;
    const compile = (kind, source) => {
      const shader = gl.createShader(kind);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        gl.deleteShader(shader);
        throw new ViewerError('The panorama renderer could not start.');
      }
      return shader;
    };
    const vertex = compile(gl.VERTEX_SHADER, VERTEX);
    const fragment = compile(gl.FRAGMENT_SHADER, FRAGMENT);
    this.program = gl.createProgram();
    gl.attachShader(this.program, vertex);
    gl.attachShader(this.program, fragment);
    gl.bindAttribLocation(this.program, 0, 'aPosition');
    gl.linkProgram(this.program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new ViewerError('The panorama renderer could not start.');
    }
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.uniforms = {};
    for (const name of ['uPanorama', 'uCenter', 'uFocal', 'uRotation']) {
      this.uniforms[name] = gl.getUniformLocation(this.program, name);
    }
    gl.useProgram(this.program);
    gl.uniform1i(this.uniforms.uPanorama, 0);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
  }

  async prepare(url) {
    if (this.current?.url === url) return this.current;
    const image = await new Promise((resolve, reject) => {
      const source = new Image();
      const finish = (ok) => {
        clearTimeout(timer);
        source.onload = null;
        source.onerror = null;
        if (ok) resolve(source);
        else { source.removeAttribute('src'); reject(new ViewerError('The panorama could not be loaded. The displayed state was retained.')); }
      };
      const timer = setTimeout(() => finish(false), 12000);
      source.onload = () => finish(true);
      source.onerror = () => finish(false);
      source.referrerPolicy = 'no-referrer';
      source.src = url;
    });
    const gl = this.gl;
    if (this.disposed || gl.isContextLost()) throw new ViewerError('The panorama renderer is unavailable. Reload the viewer.');
    const maximum = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    if (!image.naturalHeight || image.naturalWidth !== image.naturalHeight * 2 || image.naturalWidth > maximum || image.naturalHeight > maximum) {
      throw new ViewerError('The panorama must be a 2:1 image supported by this browser’s texture limit.');
    }
    const texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (gl.getError() !== gl.NO_ERROR) {
      gl.deleteTexture(texture);
      throw new ViewerError('The panorama texture could not be loaded. The displayed state was retained.');
    }
    const record = { texture, url };
    this.pending.add(record);
    return record;
  }

  discard(record) {
    if (record && record !== this.current && this.pending.delete(record)) this.gl.deleteTexture(record.texture);
  }

  show(record, camera, profile) {
    const previous = this.current;
    const previousCamera = this.lastCamera;
    const previousProfile = this.lastProfile;
    this.current = record;
    let drawn = false;
    try { drawn = this.draw(camera, profile); }
    catch { drawn = false; }
    if (!drawn) {
      this.current = previous;
      if (previous && previousCamera && previousProfile) {
        try { this.draw(previousCamera, previousProfile); }
        catch { /* A lost context is reported by the application. */ }
      }
      return false;
    }
    this.pending.delete(record);
    if (previous && previous !== record) this.gl.deleteTexture(previous.texture);
    return true;
  }

  draw(camera, profile) {
    const gl = this.gl;
    if (this.disposed || gl.isContextLost() || !this.current) return false;
    // Canvas stays in the layout while video overlays it. Never infer dimensions
    // from a hidden video, and never divide by a zero-size stage.
    const cssWidth = this.canvas.clientWidth;
    const cssHeight = this.canvas.clientHeight;
    if (!(cssWidth > 0 && cssHeight > 0)) return false;
    const ratio = Number.isFinite(window.devicePixelRatio) ? Math.min(2, Math.max(1, window.devicePixelRatio)) : 1;
    const width = Math.max(1, Math.round(cssWidth * ratio));
    const height = Math.max(1, Math.round(cssHeight * ratio));
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const fit = Math.min(width / profile.width, height / profile.height);
    const offsetX = (width - profile.width * fit) / 2;
    const offsetY = (height - profile.height * fit) / 2;
    const box = profile.content_box;
    const left = Math.round(offsetX + box[0] * fit);
    const right = Math.round(offsetX + box[2] * fit);
    const top = Math.round(offsetY + box[1] * fit);
    const bottom = Math.round(offsetY + box[3] * fit);
    const contentWidth = Math.max(1, right - left);
    const contentHeight = Math.max(1, bottom - top);
    const pitch = camera.pitch * DEG;
    const yaw = camera.yaw * DEG;
    const focal = (contentWidth / 2) / Math.tan(camera.hfov * DEG / 2);
    if (![focal, pitch, yaw].every(Number.isFinite) || focal <= 0) return false;
    gl.viewport(left, height - bottom, contentWidth, contentHeight);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.current.texture);
    gl.uniform2f(this.uniforms.uCenter, left + contentWidth / 2, height - top - contentHeight / 2);
    gl.uniform1f(this.uniforms.uFocal, focal);
    gl.uniform4f(this.uniforms.uRotation, Math.cos(pitch), Math.sin(pitch), Math.cos(yaw), Math.sin(yaw));
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR) return false;
    this.lastCamera = { ...camera };
    this.lastProfile = profile;
    return true;
  }

  destroy() {
    this.disposed = true;
    const gl = this.gl;
    for (const record of this.pending) gl.deleteTexture(record.texture);
    this.pending.clear();
    if (this.current) gl.deleteTexture(this.current.texture);
    gl.deleteBuffer(this.buffer);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.program);
    this.current = null;
  }
}
