// ITPS Maze2D — browser-side ACT + DP inference.
//
// Direct port of the unconditional mouse-follow loop from
// `itps/interact_maze2d.py` (UnconditionalMaze.run, lines 261-274) for both
// the ACT and Diffusion Policy engines. All inference runs locally via ONNX
// Runtime Web; the page never sends the agent position back to a server.
//
// Engine layout:
//   1. Maze constants & geometry      (MazeEnv coordinate system)
//   2. Math helpers                   (PRNG, Gaussian, rainbow, color blend,
//                                      DDIM scheduler for DP)
//   3. Rendering                      (Canvas2D port of MazeEnv.update_screen)
//   4. ONNX Runtime Web infrastructure (configureOrt, probeWebGpu, session
//                                      cache, build/swap)
//   5. ACT inference                  (runACT)
//   6. DP inference                   (runDP -- DDIM loop in JS, UNet via ORT)
//   7. Mouse-follow loop + engine dispatch (always-last coalescing)
//   8. Bootstrap + UI wiring (engine radio, batch slider, DDIM-steps slider)


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
const ACT_STATE_MEAN = [3.6688416004180908, 5.356582641601562];
const ACT_STATE_STD = [1.817331075668335, 2.558823823928833];
const ACT_ACTION_MEAN = [3.6688497066497803, 5.356590270996094];
const ACT_ACTION_STD = [1.8173370361328125, 2.558828830718994];

// Per-modality normalization stats for DP. Note: DP uses min_max (the wrapper
// produces actions in [-1, 1]); ACT uses mean_std. State, env_state, and
// action all share the same min/max values for Maze2D.
const DP_STATE_MIN = [0.48594850301742554, 0.5076655149459839];
const DP_STATE_MAX = [7.209974765777588, 10.214756965637207];
const DP_ACTION_MIN = [0.48594850301742554, 0.5076655149459839];
const DP_ACTION_MAX = [7.209974765777588, 10.214756965637207];

// Model shape constants.
//
// `currentBatchSize` is the number of trajectories sampled per inference call.
// The original CLI uses 32 to highlight DP's multimodality; ACT is near-
// unimodal (we measured std=0.0044 across the batch in normalized action
// space -- all 32 samples converge to nearly the same path), so 8 looks
// visually identical to 32 while running 4x faster. Exposed as a slider;
// changing it triggers a session recompile (per active engine) so ORT picks
// the optimal kernel for the new fixed batch axis.
const DEFAULT_BATCH_SIZE = 8;
let currentBatchSize = DEFAULT_BATCH_SIZE;

// ACT-specific (from configuration_act.py + checkpoint).
const ACT_CHUNK_SIZE = 64; // action prediction horizon
const LATENT_DIM = 32; // ACT VAE latent dim

// DP-specific (from configuration_diffusion.py + checkpoint).
const DP_HORIZON = 64; // action prediction horizon
const DP_N_OBS_STEPS = 2; // observations are repeated twice in time
const DP_GLOBAL_COND_DIM = 8; // (state_dim + env_state_dim) * n_obs_steps
const DP_NUM_TRAIN_TIMESTEPS = 100; // diffusion training timesteps
// Default 4 keeps DP's mouse-follow loop responsive at batch=8 on M-series
// WebGPU (~30 ms / frame, ~30 FPS). The paper's CLI uses 10 (slower but
// slightly cleaner samples), which is also the slider's upper bound.
const DEFAULT_DDIM_STEPS = 4;
let currentDdimSteps = DEFAULT_DDIM_STEPS;

// Both engines draw `CHUNK_SIZE` trajectory samples per batch. ACT and DP
// share this length (both are 64) so the same drawing code works for both.
const CHUNK_SIZE = 64;
const LATENT_SEED = 0; // matches seeded_context(0) in interact_maze2d.py:240

// Engine selection. DP is the default per the user's spec; ACT loads on toggle.
const ENGINES = ["dp", "act"];
let currentEngine = "dp";

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

// ---------- DDIM scheduler (used by runDP) ---------------------------------
//
// `ALPHAS_CUMPROD` is the diffusers `squaredcos_cap_v2` schedule for
// num_train_timesteps=100, baked at export time so we don't have to port the
// formula. (Generated by scripts/export_onnx.py --engine dp; copy-pasted here
// from its stdout.) Together with `ddimStepInPlace` this is a faithful port
// of `diffusers.DDIMScheduler.step` for prediction_type="epsilon",
// clip_sample=True (range=1.0), eta=0.
const ALPHAS_CUMPROD = new Float32Array([
  0.9993687272, 0.9982525110, 0.9966524243, 0.9945700169, 0.9920073152,
  0.9889668226, 0.9854514599, 0.9814646244, 0.9770103097, 0.9720926881,
  0.9667166471, 0.9608874321, 0.9546105862, 0.9478922486, 0.9407390356,
  0.9331577420, 0.9251558185, 0.9167410135, 0.9079215527, 0.8987058997,
  0.8891031146, 0.8791224360, 0.8687736392, 0.8580667377, 0.8470121026,
  0.8356205225, 0.8239030242, 0.8118710518, 0.7995361686, 0.7869104743,
  0.7740061879, 0.7608358264, 0.7474122047, 0.7337483168, 0.7198575139,
  0.7057532072, 0.6914491653, 0.6769592166, 0.6622974277, 0.6474781632,
  0.6325156689, 0.6174246073, 0.6022195220, 0.5869152546, 0.5715266466,
  0.5560685992, 0.5405561924, 0.5250044465, 0.5094285011, 0.4938435256,
  0.4782645702, 0.4627067745, 0.4471853077, 0.4317152202, 0.4163115025,
  0.4009891748, 0.3857631087, 0.3706480265, 0.3556586802, 0.3408096135,
  0.3261152208, 0.3115897775, 0.2972474396, 0.2831021249, 0.2691675127,
  0.2554571927, 0.2419844568, 0.2287624180, 0.2158038765, 0.2031214684,
  0.1907274723, 0.1786339432, 0.1668526232, 0.1553949714, 0.1442720890,
  0.1334948093, 0.1230735704, 0.1130185053, 0.1033393815, 0.0940456092,
  0.0851461962, 0.0766498074, 0.0685646757, 0.0608986728, 0.0536592305,
  0.0468533821, 0.0404877439, 0.0345684960, 0.0291013829, 0.0240917224,
  0.0195443742, 0.0154637592, 0.0118538374, 0.0087181171, 0.0060596438,
  0.0038809993, 0.0021842998, 0.0009711928, 0.0002428572, 0.0000002429,
]);

// Mirrors diffusers DDIMScheduler.set_timesteps for our config: evenly spaced
// descending integer timesteps. For N=10 -> [90, 80, ..., 0]. For N=4 -> [75,
// 50, 25, 0]. For N=1 -> [0]. (Verified against PyTorch.)
function ddimTimesteps(numInferenceSteps) {
  const step = Math.floor(DP_NUM_TRAIN_TIMESTEPS / numInferenceSteps);
  const ts = new Int32Array(numInferenceSteps);
  for (let i = 0; i < numInferenceSteps; i++) ts[i] = (numInferenceSteps - 1 - i) * step;
  return ts;
}

// In-place DDIM update for prediction_type="epsilon", clip_sample=True,
// eta=0 (deterministic). Matches diffusers DDIMScheduler.step exactly.
//
//   x0    = (sample - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
//   x0    = clip(x0, -1, 1)
//   prev  = sqrt(alpha_t-1) * x0 + sqrt(1 - alpha_t-1) * eps
//
// `tPrev = -1` means "this is the last step", in which case we use the
// `final_alpha_cumprod = 1.0` convention from diffusers.
function ddimStepInPlace(sample, eps, t, tPrev) {
  const aT = ALPHAS_CUMPROD[t];
  const aP = tPrev < 0 ? 1.0 : ALPHAS_CUMPROD[tPrev];
  const sqrtAT = Math.sqrt(aT);
  const sqrtOmAT = Math.sqrt(1 - aT);
  const sqrtAP = Math.sqrt(aP);
  const sqrtOmAP = Math.sqrt(1 - aP);
  for (let k = 0; k < sample.length; k++) {
    let x0 = (sample[k] - sqrtOmAT * eps[k]) / sqrtAT;
    if (x0 < -1) x0 = -1;
    else if (x0 > 1) x0 = 1;
    sample[k] = sqrtAP * x0 + sqrtOmAP * eps[k];
  }
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
// 4. ONNX Runtime Web infrastructure (engine-agnostic)
// ============================================================================

const statusEl = document.getElementById("status");
let session = null;            // active ORT session (= sessionCache[currentEngine])
let chosenEp = null;           // EP of the active session ("WebGPU" or "WASM")
let adapterDescription = "";   // human-readable WebGPU adapter name

// Cache one session per engine, plus the batchSize it was built with. If the
// active batch matches the cached session's batch, switching engines is
// instant; otherwise the engine activation rebuilds.
const sessionCache = new Map(); // engine -> { session, batchSize, ep }

// `inferenceLocked` blocks `tick()` from starting a new run during a session
// swap (engine or batch-size change). Existing in-flight runs drain naturally
// because the swap waits for `pending` to clear before committing.
let inferenceLocked = false;

// Per-engine ONNX file metadata used in status messages.
const ENGINE_FILES = {
  act: { onnx: "act.onnx", sizeMb: 44 },
  dp:  { onnx: "dp_unet.onnx", sizeMb: 31 },
};

function configureOrt() {
  if (typeof navigator !== "undefined" && navigator.hardwareConcurrency) {
    ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency, 8);
  }
  ort.env.wasm.simd = true;
  ort.env.logLevel = "warning";
}

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

// Build (and freeze the batch axis on) a fresh session for one engine.
async function buildSession(engine, batchSize) {
  const sessionOpts = {
    graphOptimizationLevel: "all",
    freeDimensionOverrides: { batch: batchSize },
  };
  const file = ENGINE_FILES[engine].onnx;
  let s, ep;
  try {
    s = await ort.InferenceSession.create(file, {
      ...sessionOpts,
      executionProviders: ["webgpu"],
      preferredOutputLocation: "cpu",
    });
    ep = "WebGPU";
  } catch (errGpu) {
    console.warn(`WebGPU session for ${engine} failed, falling back to WASM:`, errGpu);
    s = await ort.InferenceSession.create(file, {
      ...sessionOpts,
      executionProviders: ["wasm"],
    });
    ep = "WASM";
  }
  return { session: s, ep };
}

// Make `engine` the active engine, building/caching its session if needed and
// pre-warming on first build. Used both for the initial load and engine-switch.
async function activateEngine(engine, batchSize) {
  allocateBuffers(engine, batchSize);
  const cached = sessionCache.get(engine);
  if (cached && cached.batchSize === batchSize) {
    session = cached.session;
    chosenEp = cached.ep;
    return false; // not freshly built; no warm-up needed
  }
  const t0 = performance.now();
  const { session: newSess, ep } = await buildSession(engine, batchSize);
  sessionCache.set(engine, { session: newSess, batchSize, ep });
  session = newSess;
  chosenEp = ep;
  console.info(
    `Built ${engine} session (batch=${batchSize}, ${ep}) in ${(performance.now() - t0).toFixed(0)} ms`,
  );
  // Pre-warm: WebGPU JIT-compiles per static shape; two warmup runs stabilize
  // the JIT caches so the user-visible FPS counter shows steady-state numbers.
  const warmT0 = performance.now();
  for (let i = 0; i < 2; i++) await runEngine([0, 0]);
  console.info(`  pre-warm: 2 inferences in ${(performance.now() - warmT0).toFixed(0)} ms`);
  return true;
}

function statusReady() {
  const adapter = adapterDescription ? ` (${adapterDescription})` : "";
  const dpExtra = currentEngine === "dp" ? ` · ${currentDdimSteps} DDIM steps` : "";
  statusEl.textContent =
    `Ready · ${chosenEp}${adapter} · ${currentEngine.toUpperCase()}${dpExtra}` +
    ` · batch=${currentBatchSize} · move your mouse over the maze`;
}

async function loadModel() {
  configureOrt();
  const file = ENGINE_FILES[currentEngine];
  statusEl.textContent = `Loading ${file.onnx} (~${file.sizeMb} MB on first visit, cached after)…`;

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

  await activateEngine(currentEngine, currentBatchSize);
  _frameTimes.length = 0;
  statusReady();
}

// Slider release handler: rebuild the active engine's session at the new batch.
async function setBatchSize(newSize) {
  if (newSize === currentBatchSize) return;

  inferenceLocked = true;
  while (pending) await new Promise((r) => setTimeout(r, 5));

  statusEl.textContent = `Recompiling ${currentEngine.toUpperCase()} for batch=${newSize}…`;
  try {
    // Build new session first; only commit state if it succeeds. (Caches for
    // *other* engines are left intact -- they're at the old batch and will be
    // rebuilt when the user next switches to them.)
    const { session: newSess, ep } = await buildSession(currentEngine, newSize);
    currentBatchSize = newSize;
    allocateBuffers(currentEngine, newSize);
    session = newSess;
    chosenEp = ep;
    sessionCache.set(currentEngine, { session: newSess, batchSize: newSize, ep });
    for (let i = 0; i < 2; i++) await runEngine([0, 0]);
    _frameTimes.length = 0;
    statusReady();
  } catch (err) {
    statusEl.textContent = `Recompile failed: ${err.message}`;
    console.error(err);
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

// Engine-radio change handler: switch active engine, lazy-loading on first use.
async function setEngine(newEngine) {
  if (newEngine === currentEngine) return;
  if (!ENGINES.includes(newEngine)) return;

  inferenceLocked = true;
  while (pending) await new Promise((r) => setTimeout(r, 5));

  const file = ENGINE_FILES[newEngine];
  const cached = sessionCache.get(newEngine);
  statusEl.textContent = cached && cached.batchSize === currentBatchSize
    ? `Switching to ${newEngine.toUpperCase()}…`
    : `Loading ${file.onnx}…`;
  try {
    const prevEngine = currentEngine;
    currentEngine = newEngine;
    try {
      await activateEngine(newEngine, currentBatchSize);
    } catch (err) {
      currentEngine = prevEngine; // revert on failure
      throw err;
    }
    _frameTimes.length = 0;
    statusReady();
    updateEngineSpecificUi();
  } catch (err) {
    statusEl.textContent = `Engine swap failed: ${err.message}`;
    console.error(err);
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


// ============================================================================
// 5. ACT inference (existing path)
// ============================================================================

let _actBufs = null; // { state, envState, latent }, all Float32Array
let _dpBufs = null;  // { sample, timestep, globalCond }, see runDP

function allocateBuffers(engine, batchSize) {
  if (engine === "act") {
    _actBufs = {
      state:    new Float32Array(batchSize * 2),
      envState: new Float32Array(batchSize * 2),
      latent:   new Float32Array(batchSize * LATENT_DIM),
    };
  } else {
    _dpBufs = {
      sample:     new Float32Array(batchSize * DP_HORIZON * 2),
      timestep:   new BigInt64Array(batchSize), // ORT int64 tensor expects BigInt64Array
      globalCond: new Float32Array(batchSize * DP_GLOBAL_COND_DIM),
    };
  }
}

async function runACT(stateXY) {
  const B = currentBatchSize;
  const bufs = _actBufs;
  // Normalize agent xy and broadcast across the batch -- mirrors einops.repeat
  // in interact_maze2d.py:229-234 (every batch element sees the same state;
  // multimodality comes from the per-element latent vector).
  const xn = (stateXY[0] - ACT_STATE_MEAN[0]) / (ACT_STATE_STD[0] + 1e-8);
  const yn = (stateXY[1] - ACT_STATE_MEAN[1]) / (ACT_STATE_STD[1] + 1e-8);
  for (let b = 0; b < B; b++) {
    bufs.state[b * 2] = xn;
    bufs.state[b * 2 + 1] = yn;
    bufs.envState[b * 2] = xn;
    bufs.envState[b * 2 + 1] = yn;
  }
  // Reseed the latent every call so the same agent xy always yields the same
  // batch of trajectories -- matches `seeded_context(0)` in the original CLI.
  fillStandardNormal(bufs.latent, LATENT_SEED);

  const feeds = {
    state:     new ort.Tensor("float32", bufs.state,    [B, 2]),
    env_state: new ort.Tensor("float32", bufs.envState, [B, 2]),
    latent:    new ort.Tensor("float32", bufs.latent,   [B, LATENT_DIM]),
  };
  const out = await session.run(feeds);
  const actionsNorm = out.actions.data; // Float32Array(B * 64 * 2)

  // Unnormalize back to maze-space xy into a fresh array so the caller can
  // hold onto it across rAF without us overwriting it on the next call.
  const actions = new Float32Array(actionsNorm.length);
  for (let i = 0; i < actionsNorm.length; i += 2) {
    actions[i]     = actionsNorm[i]     * ACT_ACTION_STD[0] + ACT_ACTION_MEAN[0];
    actions[i + 1] = actionsNorm[i + 1] * ACT_ACTION_STD[1] + ACT_ACTION_MEAN[1];
  }
  return actions;
}


// ============================================================================
// 6. DP inference (DDIM loop in JS, UNet via ORT)
// ============================================================================

async function runDP(stateXY) {
  const B = currentBatchSize;
  const numSteps = currentDdimSteps;
  const bufs = _dpBufs;

  // 1. min_max normalize state to [-1, 1]: x_norm = ((x - min) / range) * 2 - 1
  const xn = ((stateXY[0] - DP_STATE_MIN[0]) / (DP_STATE_MAX[0] - DP_STATE_MIN[0] + 1e-8)) * 2 - 1;
  const yn = ((stateXY[1] - DP_STATE_MIN[1]) / (DP_STATE_MAX[1] - DP_STATE_MIN[1] + 1e-8)) * 2 - 1;

  // 2. Build global_cond: concat([state, env_state], dim=-1).flatten(1).
  // For Maze2D with n_obs_steps=2, state == env_state == agent xy at every
  // timestep; the same 8-vector goes into every batch element.
  for (let b = 0; b < B; b++) {
    const off = b * DP_GLOBAL_COND_DIM;
    bufs.globalCond[off]     = xn; bufs.globalCond[off + 1] = yn;
    bufs.globalCond[off + 2] = xn; bufs.globalCond[off + 3] = yn;
    bufs.globalCond[off + 4] = xn; bufs.globalCond[off + 5] = yn;
    bufs.globalCond[off + 6] = xn; bufs.globalCond[off + 7] = yn;
  }

  // 3. Initial sample ~ N(0, I). Reseed each call so the same agent xy always
  //    produces the same batch of trajectories (matches the original CLI's
  //    `seeded_context(0)`).
  fillStandardNormal(bufs.sample, LATENT_SEED);

  // 4. DDIM denoising loop. The UNet is the only ORT call; the scheduler step
  //    runs in JS. See the diffusers parity check in
  //    scripts/verify_dp_js_math.py -- both paths agree to 0.0e+00.
  const timesteps = ddimTimesteps(numSteps);
  for (let i = 0; i < timesteps.length; i++) {
    const t = timesteps[i];
    const tPrev = i + 1 < timesteps.length ? timesteps[i + 1] : -1;

    const tBig = BigInt(t);
    for (let b = 0; b < B; b++) bufs.timestep[b] = tBig;

    const feeds = {
      sample:      new ort.Tensor("float32", bufs.sample,     [B, DP_HORIZON, 2]),
      timestep:    new ort.Tensor("int64",   bufs.timestep,   [B]),
      global_cond: new ort.Tensor("float32", bufs.globalCond, [B, DP_GLOBAL_COND_DIM]),
    };
    const out = await session.run(feeds);
    ddimStepInPlace(bufs.sample, out.noise_pred.data, t, tPrev);
  }

  // 5. Unnormalize: actions in [-1, 1] -> maze coords.
  // Original CLI slices to actions[:, n_obs_steps - 1 : ...] but for the
  // visualization the very-first-step difference is imperceptible -- we keep
  // all 64 to match the chunk size used by the shared rendering code.
  const flatLen = B * DP_HORIZON * 2;
  const actions = new Float32Array(flatLen);
  const rangeX = DP_ACTION_MAX[0] - DP_ACTION_MIN[0];
  const rangeY = DP_ACTION_MAX[1] - DP_ACTION_MIN[1];
  for (let i = 0; i < flatLen; i += 2) {
    actions[i]     = (bufs.sample[i]     + 1) * 0.5 * rangeX + DP_ACTION_MIN[0];
    actions[i + 1] = (bufs.sample[i + 1] + 1) * 0.5 * rangeY + DP_ACTION_MIN[1];
  }
  return actions;
}

async function runEngine(stateXY) {
  return currentEngine === "act" ? runACT(stateXY) : runDP(stateXY);
}


// ============================================================================
// 7. Mouse-follow loop with always-last coalescing
// ============================================================================
//
// `latestXY` always holds the *most recent* mousemove position. While inference
// is running, additional mousemove events overwrite it instead of queuing.
// When `tick()` finishes one inference + render, it loops back to consume the
// latest position. JS equivalent of Gradio's `trigger_mode="always_last"`.

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
      const actions = await runEngine([mx, my]);
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
        const dpExtra = currentEngine === "dp" ? ` · ${currentDdimSteps} steps` : "";
        statusEl.textContent =
          `${chosenEp} · ${currentEngine.toUpperCase()}${dpExtra}` +
          ` · batch=${currentBatchSize} · ${stat.ms.toFixed(0)} ms / frame · ${stat.fps.toFixed(1)} FPS`;
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
// 8. Bootstrap + UI wiring
// ============================================================================

function updateEngineSpecificUi() {
  // Toggle DDIM-steps control visibility based on active engine.
  const controls = document.getElementById("controls");
  if (controls) controls.dataset.engine = currentEngine;
}

function wireBatchSlider() {
  const slider = document.getElementById("batch-slider");
  const valueEl = document.getElementById("batch-value");
  if (!slider || !valueEl) return;
  slider.value = String(currentBatchSize);
  valueEl.textContent = String(currentBatchSize);

  // Live-update the displayed value while dragging, but only recompile on
  // release (`change` event) -- avoids a recompile per pixel of slider travel.
  slider.addEventListener("input", () => {
    valueEl.textContent = slider.value;
  });
  slider.addEventListener("change", async () => {
    const n = parseInt(slider.value, 10);
    if (!Number.isFinite(n) || n < 1 || n > 32) return;
    await setBatchSize(n);
  });
}

function wireStepsSlider() {
  const slider = document.getElementById("steps-slider");
  const valueEl = document.getElementById("steps-value");
  if (!slider || !valueEl) return;
  slider.value = String(currentDdimSteps);
  valueEl.textContent = String(currentDdimSteps);

  // Both `input` and `change` just update `currentDdimSteps`. No session
  // recompile is needed because the steps count controls the JS DDIM loop
  // length, not the ONNX UNet shape -- the next runDP call picks up the new
  // value automatically.
  const onChange = () => {
    const n = parseInt(slider.value, 10);
    if (!Number.isFinite(n) || n < 1 || n > 10) return;
    currentDdimSteps = n;
    valueEl.textContent = String(n);
    if (currentEngine === "dp") statusReady();
  };
  slider.addEventListener("input", onChange);
  slider.addEventListener("change", onChange);
}

function wireEngineRadio() {
  const radios = document.querySelectorAll('input[name="engine"]');
  radios.forEach((r) => {
    r.checked = r.value === currentEngine;
    r.addEventListener("change", async () => {
      if (!r.checked) return;
      await setEngine(r.value);
    });
  });
}

async function main() {
  // Initial render: empty maze with agent at the canvas center.
  const [cgx, cgy] = xy2gui(MAZE_ROWS / 2 - OFFSET, MAZE_COLS / 2 - OFFSET);
  renderFrame(cgx, cgy, false, null, null);

  wireBatchSlider();
  wireStepsSlider();
  wireEngineRadio();
  updateEngineSpecificUi();

  await loadModel();
  canvas.addEventListener("mousemove", onMouseMove);
}

main().catch((err) => {
  statusEl.textContent = `Error: ${err.message}`;
  console.error(err);
});
