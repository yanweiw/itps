// ITPS Maze2D — browser-side ACT inference.
//
// Direct port of the unconditional mouse-follow loop from
// `itps/interact_maze2d.py` (UnconditionalMaze.run, lines 261-274).
// All inference runs locally via ONNX Runtime Web; the page never sends
// the agent position back to a server.
//
// Layout matches the Plan in browser-act-static-demo:
//   1. Maze constants & geometry      (MazeEnv coordinate system)
//   2. Math helpers                   (PRNG, Gaussian, rainbow, color blend)
//   3. Rendering                      (Canvas2D port of MazeEnv.update_screen)
//   4. ONNX Runtime Web inference     (loadModel + runACT)
//   5. Mouse-follow loop              (always-last coalescing)
//   6. Bootstrap


// ============================================================================
// 1. Maze constants & geometry
// ============================================================================

// 9 rows x 12 cols binary maze. 1 = wall, 0 = free.
// Verbatim from itps/interact_maze2d.py:74-82.
const MAZE = [
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
  [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
  [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
  [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
  [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];
const MAZE_ROWS = 9; // x axis in maze space
const MAZE_COLS = 12; // y axis in maze space
const GUI_W = 1200;
const GUI_H = 900;
const CELL_W = GUI_W / MAZE_COLS; // 100 px per maze cell, horizontally
const CELL_H = GUI_H / MAZE_ROWS; // 100 px per maze cell, vertically
const OFFSET = 0.5; // centers (x, y) inside its cell

// Per-modality normalization stats; read from the trained ACT checkpoint.
// State and env_state share stats (and they share values too in Maze2D).
const STATE_MEAN = [3.6688416004180908, 5.356582641601562];
const STATE_STD = [1.817331075668335, 2.558823823928833];
const ACTION_MEAN = [3.6688497066497803, 5.356590270996094];
const ACTION_STD = [1.8173370361328125, 2.558828830718994];

// Model shape constants (from configuration_act.py + checkpoint).
//
// `currentBatchSize` is the number of trajectories sampled per inference call.
// The original CLI uses 32 to highlight diffusion policy's multimodality. ACT
// is near-unimodal (we measured std=0.0044 across the batch in normalized
// action space -- all 32 samples converge to nearly the same path), so 8
// trajectories look visually identical to 32 while running 4x faster.
// Exposed as a slider in the UI; changing it triggers a session recompile so
// ORT can pick the optimal kernel for the new fixed batch axis.
const DEFAULT_BATCH_SIZE = 8;
let currentBatchSize = DEFAULT_BATCH_SIZE;
const CHUNK_SIZE = 64; // action prediction horizon
const LATENT_DIM = 32; // VAE latent dim
const LATENT_SEED = 0; // matches seeded_context(0) in interact_maze2d.py:240

// xy2gui / gui2xy: port of MazeEnv.xy2gui / MazeEnv.gui2xy (lines 134-143).
// Note the axis swap: maze x (rows) maps to gui y (vertical), maze y (cols)
// maps to gui x (horizontal).
function xy2gui(x, y) {
  return [(y + OFFSET) * CELL_W, (x + OFFSET) * CELL_H];
}
function gui2xy(gx, gy) {
  return [(gy / GUI_H) * MAZE_ROWS - OFFSET, (gx / GUI_W) * MAZE_COLS - OFFSET];
}

// Port of MazeEnv.check_collision. Returns Array<bool> of length `batchSize`.
// `xyTrajFlat` is interleaved Float32Array(B * S * 2) of (x, y) maze coords.
function checkCollision(xyTrajFlat, batchSize, numSteps) {
  const collisions = new Array(batchSize).fill(false);
  for (let b = 0; b < batchSize; b++) {
    for (let s = 0; s < numSteps; s++) {
      const idx = (b * numSteps + s) * 2;
      let x = xyTrajFlat[idx];
      let y = xyTrajFlat[idx + 1];
      if (x < 0) x = 0;
      else if (x > MAZE_ROWS - 1) x = MAZE_ROWS - 1;
      if (y < 0) y = 0;
      else if (y > MAZE_COLS - 1) y = MAZE_COLS - 1;
      const mx = Math.round(x);
      const my = Math.round(y);
      if (MAZE[mx][my]) {
        collisions[b] = true;
        break;
      }
    }
  }
  return collisions;
}


// ============================================================================
// 2. Math helpers
// ============================================================================

// Matplotlib `rainbow` colormap sampled at 64 evenly spaced points in [0, 1].
// Generated with: plt.get_cmap('rainbow')(np.linspace(0, 1, 64)) * 255.
const RAINBOW_64 = [
  [127, 0, 255], [119, 12, 254], [111, 25, 254], [103, 37, 254],
  [95, 49, 253], [87, 62, 253], [79, 74, 252], [71, 86, 251],
  [63, 97, 250], [55, 109, 248], [47, 120, 247], [39, 131, 245],
  [31, 142, 243], [23, 152, 242], [15, 162, 239], [7, 171, 237],
  [2, 183, 234], [10, 191, 232], [18, 199, 229], [26, 207, 226],
  [34, 214, 223], [42, 220, 220], [50, 226, 217], [58, 232, 214],
  [66, 237, 210], [74, 241, 207], [82, 245, 203], [90, 248, 199],
  [98, 250, 195], [106, 252, 191], [114, 254, 187], [122, 254, 183],
  [132, 254, 177], [140, 254, 172], [148, 252, 168], [156, 250, 163],
  [164, 248, 158], [172, 245, 153], [180, 241, 148], [188, 237, 143],
  [196, 232, 138], [204, 226, 132], [212, 220, 127], [220, 214, 122],
  [228, 207, 116], [236, 199, 110], [244, 191, 105], [252, 183, 99],
  [255, 171, 92], [255, 162, 86], [255, 152, 80], [255, 142, 74],
  [255, 131, 68], [255, 120, 62], [255, 109, 56], [255, 97, 49],
  [255, 86, 43], [255, 74, 37], [255, 62, 31], [255, 49, 25],
  [255, 37, 18], [255, 25, 12], [255, 12, 6], [255, 0, 0],
];

// mulberry32: tiny seedable PRNG. We only need it to reproduce the multimodal
// sample set per agent position, not for cryptographic anything.
function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller transform: turn uniform PRNG output into standard-normal samples.
function gaussianSampler(rng) {
  let cached = null;
  return () => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return v;
    }
    let u;
    do {
      u = rng();
    } while (u <= 0);
    const v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    const ang = 2 * Math.PI * v;
    cached = mag * Math.sin(ang);
    return mag * Math.cos(ang);
  };
}

function fillStandardNormal(arr, seed) {
  const rng = mulberry32(seed);
  const sample = gaussianSampler(rng);
  for (let i = 0; i < arr.length; i++) arr[i] = sample();
}

function blendWhite(rgb, factor) {
  return [
    Math.round((1 - factor) * rgb[0] + factor * 255),
    Math.round((1 - factor) * rgb[1] + factor * 255),
    Math.round((1 - factor) * rgb[2] + factor * 255),
  ];
}


// ============================================================================
// 3. Rendering (port of MazeEnv.update_screen, lines 156-188)
// ============================================================================

const canvas = document.getElementById("maze");
const ctx = canvas.getContext("2d");

// Cache the maze background to an offscreen canvas: it's static, so we only
// build it once and blit it per frame.
let _mazeBg = null;
function getMazeBackground() {
  if (_mazeBg) return _mazeBg;
  _mazeBg = document.createElement("canvas");
  _mazeBg.width = GUI_W;
  _mazeBg.height = GUI_H;
  const bg = _mazeBg.getContext("2d");
  bg.fillStyle = "#ffffff";
  bg.fillRect(0, 0, GUI_W, GUI_H);
  bg.fillStyle = "#000000";
  for (let i = 0; i < MAZE_ROWS; i++) {
    for (let j = 0; j < MAZE_COLS; j++) {
      if (MAZE[i][j]) bg.fillRect(j * CELL_W, i * CELL_H, CELL_W, CELL_H);
    }
  }
  return _mazeBg;
}

function drawMaze() {
  ctx.drawImage(getMazeBackground(), 0, 0);
}

function drawTrajectories(xyTrajFlat, collisions) {
  // For each batch element, draw chunk_size circles colored by rainbow time.
  // Trajectories that collide get tinted toward white (factor=0.8). Faithful
  // to the original CLI: trajectories are drawn at the model's raw action
  // coords, which can leave a visible gap between the agent and the start of
  // the rainbow when ACT's predicted action[0] differs from the current state.
  const B = currentBatchSize;
  for (let b = 0; b < B; b++) {
    const factor = collisions[b] ? 0.8 : 0.0;
    for (let s = 0; s < CHUNK_SIZE - 1; s++) {
      const idx = (b * CHUNK_SIZE + s) * 2;
      const [gx, gy] = xy2gui(xyTrajFlat[idx], xyTrajFlat[idx + 1]);
      const [r, g, bl] = blendWhite(RAINBOW_64[s], factor);
      ctx.fillStyle = `rgb(${r},${g},${bl})`;
      ctx.beginPath();
      ctx.arc(gx, gy, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawAgent(gx, gy, inCollision) {
  const rgb = inCollision ? blendWhite([255, 0, 0], 0.8) : [255, 0, 0];
  ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  ctx.beginPath();
  ctx.arc(gx, gy, 20, 0, Math.PI * 2);
  ctx.fill();
}

function renderFrame(agentGx, agentGy, agentInCollision, xyTrajFlat, collisions) {
  drawMaze();
  if (xyTrajFlat) drawTrajectories(xyTrajFlat, collisions);
  drawAgent(agentGx, agentGy, agentInCollision);
}


// ============================================================================
// 4. ONNX Runtime Web inference
// ============================================================================

const statusEl = document.getElementById("status");
let session = null;
let chosenEp = null;
let adapterDescription = "";

// Configure the ORT runtime *before* creating any sessions. Defaults are
// usually fine, but explicit is better for diagnostics.
function configureOrt() {
  // Use as many WASM threads as the host has cores. Requires SharedArrayBuffer,
  // which requires the page to be served with COOP/COEP headers (see
  // scripts/serve.py); ORT silently falls back to single-threaded otherwise.
  if (typeof navigator !== "undefined" && navigator.hardwareConcurrency) {
    ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency, 8);
  }
  ort.env.wasm.simd = true;
  ort.env.logLevel = "warning";
}

// Inspect the WebGPU adapter so the status pill can name the actual GPU.
async function probeWebGpu() {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
    return info;
  } catch (err) {
    console.warn("WebGPU probe failed:", err);
    return null;
  }
}

// `inferenceLocked` blocks `tick()` from starting a new run during a session
// swap (batch-size change). Existing in-flight runs drain naturally because
// `setBatchSize` waits for `pending` to clear before swapping.
let inferenceLocked = false;

// Build (and pre-warm) a fresh ORT session locked to `batchSize`.
async function buildSession(batchSize) {
  // freeDimensionOverrides locks the dynamic 'batch' axis to a known size, so
  // ORT can compile fully-static-shape kernels at session-create time instead
  // of falling back to slower generic-batch kernels at run time.
  const sessionOpts = {
    graphOptimizationLevel: "all",
    freeDimensionOverrides: { batch: batchSize },
  };
  let s;
  try {
    s = await ort.InferenceSession.create("act.onnx", {
      ...sessionOpts,
      executionProviders: ["webgpu"],
      preferredOutputLocation: "cpu",
    });
    chosenEp = "WebGPU";
  } catch (errGpu) {
    console.warn("WebGPU session failed, falling back to WASM:", errGpu);
    s = await ort.InferenceSession.create("act.onnx", {
      ...sessionOpts,
      executionProviders: ["wasm"],
    });
    chosenEp = "WASM";
  }
  return s;
}

function statusReady() {
  const adapter = adapterDescription ? ` (${adapterDescription})` : "";
  statusEl.textContent =
    `Ready · ${chosenEp}${adapter} · batch=${currentBatchSize} · move your mouse over the maze`;
}

async function loadModel() {
  configureOrt();
  statusEl.textContent = "Loading act.onnx (~44 MB on first visit, cached after)…";

  const adapterInfo = await probeWebGpu();
  if (adapterInfo) {
    const parts = [adapterInfo.vendor, adapterInfo.architecture, adapterInfo.device, adapterInfo.description]
      .filter(Boolean)
      .join(" ");
    adapterDescription = parts || "unknown adapter";
    console.info("WebGPU adapter:", adapterInfo);
  } else {
    console.info("WebGPU adapter: not available, will fall back to WASM");
  }

  allocateBuffers(currentBatchSize);

  const t0 = performance.now();
  session = await buildSession(currentBatchSize);
  const loadDt = ((performance.now() - t0) / 1000).toFixed(1);
  console.info(`ORT session ready in ${loadDt}s on ${chosenEp}`);

  // Pre-warm: run a few dummy inferences before the user touches anything.
  // The first WebGPU call pays kernel-compilation cost (often 0.3-3 s on a
  // small model like ACT); two more stabilize JIT caches. Without this, the
  // FPS counter would be skewed by warm-up noise for the first few seconds
  // of mouse motion.
  statusEl.textContent = "Compiling GPU kernels (one-time)…";
  const warmT0 = performance.now();
  for (let i = 0; i < 3; i++) await runACT([0, 0]);
  console.info(`Pre-warm: 3 inferences in ${(performance.now() - warmT0).toFixed(0)} ms`);

  _frameTimes.length = 0;
  statusReady();
}

// Swap to a new fixed-batch session. Called from the slider's `change` event.
async function setBatchSize(newSize) {
  if (newSize === currentBatchSize) return;

  // Block new inferences and wait for any in-flight tick to finish.
  inferenceLocked = true;
  while (pending) await new Promise((r) => setTimeout(r, 5));

  statusEl.textContent = `Recompiling for batch=${newSize}…`;
  try {
    const t0 = performance.now();
    // Build the new session first; only commit state if it succeeds. This
    // way a failed recompile leaves the runtime in its previous working
    // state rather than half-swapped.
    const newSession = await buildSession(newSize);
    currentBatchSize = newSize;
    allocateBuffers(newSize);
    session = newSession;
    // Pre-warm new session (WebGPU JIT-compiles per static shape).
    await runACT([0, 0]);
    await runACT([0, 0]);
    console.info(`Switched to batch=${newSize} in ${(performance.now() - t0).toFixed(0)} ms`);
    _frameTimes.length = 0;
    statusReady();
  } catch (err) {
    statusEl.textContent = `Recompile failed: ${err.message}`;
    console.error(err);
    // currentBatchSize / buffers / session are unchanged on failure.
  } finally {
    inferenceLocked = false;
    if (latestXY && !pending) {
      tick().catch((err) => {
        statusEl.textContent = `Inference error: ${err.message}`;
        console.error(err);
      });
    }
  }
}

// Rolling FPS estimate over the last few inference calls.
const _frameTimes = [];
function recordFrameTime(ms) {
  _frameTimes.push(ms);
  if (_frameTimes.length > 30) _frameTimes.shift();
  const avg = _frameTimes.reduce((a, b) => a + b, 0) / _frameTimes.length;
  return { ms: avg, fps: 1000 / avg };
}

// Pre-allocated input buffers; re-allocated whenever the batch size changes.
let _stateBuf = null;
let _envStateBuf = null;
let _latentBuf = null;

function allocateBuffers(batchSize) {
  _stateBuf = new Float32Array(batchSize * 2);
  _envStateBuf = new Float32Array(batchSize * 2);
  _latentBuf = new Float32Array(batchSize * LATENT_DIM);
}

async function runACT(stateXY) {
  const B = currentBatchSize;
  // Normalize agent xy and broadcast across the batch -- mirrors einops.repeat
  // in interact_maze2d.py:229-234 (every batch element sees the same state;
  // multimodality comes from the per-element latent vector).
  const xn = (stateXY[0] - STATE_MEAN[0]) / (STATE_STD[0] + 1e-8);
  const yn = (stateXY[1] - STATE_MEAN[1]) / (STATE_STD[1] + 1e-8);
  for (let b = 0; b < B; b++) {
    _stateBuf[b * 2] = xn;
    _stateBuf[b * 2 + 1] = yn;
    _envStateBuf[b * 2] = xn;
    _envStateBuf[b * 2 + 1] = yn;
  }
  // Reseed the latent every call so the same agent xy always yields the same
  // batch of trajectories -- matches `seeded_context(0)` in the original CLI.
  fillStandardNormal(_latentBuf, LATENT_SEED);

  const feeds = {
    state: new ort.Tensor("float32", _stateBuf, [B, 2]),
    env_state: new ort.Tensor("float32", _envStateBuf, [B, 2]),
    latent: new ort.Tensor("float32", _latentBuf, [B, LATENT_DIM]),
  };
  const out = await session.run(feeds);
  const actionsNorm = out.actions.data; // Float32Array(B * CHUNK_SIZE * 2)

  // Unnormalize back to maze-space xy into a fresh array so the caller can
  // hold onto it across rAF without us overwriting it on the next call.
  const actions = new Float32Array(actionsNorm.length);
  for (let i = 0; i < actionsNorm.length; i += 2) {
    actions[i] = actionsNorm[i] * ACTION_STD[0] + ACTION_MEAN[0];
    actions[i + 1] = actionsNorm[i + 1] * ACTION_STD[1] + ACTION_MEAN[1];
  }
  return actions;
}


// ============================================================================
// 5. Mouse-follow loop with always-last coalescing
// ============================================================================
//
// `latestXY` always holds the *most recent* mousemove position. While inference
// is running, additional mousemove events overwrite it instead of queuing.
// When `tick()` finishes one inference + render, it loops back to consume the
// latest position. This is the JS equivalent of Gradio's
// `trigger_mode="always_last"`: the user never sees stale predictions for an
// old mouse position, just the most recent one the device could keep up with.

let latestXY = null; // (gui_x, gui_y) of the most recent mousemove
let pending = false; // true while tick() is running

function clientToCanvas(e) {
  const rect = canvas.getBoundingClientRect();
  return [
    (e.clientX - rect.left) * (canvas.width / rect.width),
    (e.clientY - rect.top) * (canvas.height / rect.height),
  ];
}

async function tick() {
  pending = true;
  try {
    while (latestXY && !inferenceLocked) {
      const [gx, gy] = latestXY;
      latestXY = null;
      const [mx, my] = gui2xy(gx, gy);

      const t0 = performance.now();
      const actions = await runACT([mx, my]);
      const collisions = checkCollision(actions, currentBatchSize, CHUNK_SIZE);
      const agentInCollision = checkCollision(
        new Float32Array([mx, my]),
        1,
        1,
      )[0];
      const dt = performance.now() - t0;
      const stat = recordFrameTime(dt);

      requestAnimationFrame(() => {
        renderFrame(gx, gy, agentInCollision, actions, collisions);
        statusEl.textContent =
          `${chosenEp} · batch=${currentBatchSize} · ` +
          `${stat.ms.toFixed(0)} ms / frame · ${stat.fps.toFixed(1)} FPS`;
      });
    }
  } finally {
    pending = false;
  }
}

function onMouseMove(e) {
  latestXY = clientToCanvas(e);
  if (!pending && !inferenceLocked) {
    tick().catch((err) => {
      statusEl.textContent = `Inference error: ${err.message}`;
      console.error(err);
    });
  }
}


// ============================================================================
// 6. Bootstrap
// ============================================================================

function wireBatchSlider() {
  const slider = document.getElementById("batch-slider");
  const valueEl = document.getElementById("batch-value");
  if (!slider || !valueEl) return;
  slider.value = String(currentBatchSize);
  valueEl.textContent = String(currentBatchSize);

  // Live-update the displayed value while dragging, but only recompile on
  // release (`change` event). This avoids triggering a recompile per pixel.
  slider.addEventListener("input", () => {
    valueEl.textContent = slider.value;
  });
  slider.addEventListener("change", async () => {
    const n = parseInt(slider.value, 10);
    if (!Number.isFinite(n) || n < 1 || n > 32) return;
    await setBatchSize(n);
  });
}

async function main() {
  // Initial render: empty maze with agent at the canvas center.
  const [cgx, cgy] = xy2gui(MAZE_ROWS / 2 - OFFSET, MAZE_COLS / 2 - OFFSET);
  renderFrame(cgx, cgy, false, null, null);
  wireBatchSlider();
  await loadModel();
  canvas.addEventListener("mousemove", onMouseMove);
}

main().catch((err) => {
  statusEl.textContent = `Error: ${err.message}`;
  console.error(err);
});
