let appState = {
  running: false,
  funcId: 'rastrigin',
  particles: 10,
  amplitude: 15.0,
  steps: 2000,
  speed: 2,
  showNoise: true,
  showTrails: true,
  showOptimum: true,
  seedOffset: 0
};

let rffField, seismicEngine, saEngine;
let seismicRenderer, saRenderer, chartRenderer;
let animationId;

const elements = {
  seismicCanvas: document.getElementById('seismicCanvas'),
  saCanvas: document.getElementById('saCanvas'),
  chartCanvas: document.getElementById('chartCanvas'),
  btnPlay: document.getElementById('btnPlay'),
  btnPause: document.getElementById('btnPause'),
  btnStep: document.getElementById('btnStep'),
  btnReset: document.getElementById('btnReset'),
  speedSlider: document.getElementById('speedSlider'),
  funcSelect: document.getElementById('funcSelect'),
  particlesInput: document.getElementById('particlesInput'),
  ampInput: document.getElementById('ampInput'),
  stepsInput: document.getElementById('stepsInput'),
  chkNoise: document.getElementById('chkNoise'),
  chkTrails: document.getElementById('chkTrails'),
  chkOptimum: document.getElementById('chkOptimum'),
  lblStep: document.getElementById('lblStep'),
  lblBest: document.getElementById('lblBest'),
  lblAmp: document.getElementById('lblAmp'),
  lblCycle: document.getElementById('lblCycle'),
  progressFill: document.getElementById('progressFill')
};

function init() {
  const fnConfig = FUNCTIONS[appState.funcId];
  elements.ampInput.value = fnConfig.recommendedAmplitude;
  appState.amplitude = fnConfig.recommendedAmplitude;
  
  rffField = new RFFField();
  
  appState.steps = parseInt(elements.stepsInput.value) || 2000;

  seismicEngine = new SeismicEngine(fnConfig, rffField, {
    nParticles: appState.particles,
    noiseAmplitude: appState.amplitude,
    nSteps: appState.steps,
  });
  
  saEngine = new SAEngine(fnConfig, {
    stepSize: fnConfig.range * 0.1,
    nSteps: appState.steps,
  });
  
  seismicEngine.reset(appState.seedOffset);
  saEngine.reset(appState.seedOffset);
  
  seismicRenderer = new Renderer(elements.seismicCanvas, fnConfig);
  saRenderer = new Renderer(elements.saCanvas, fnConfig);
  chartRenderer = new ChartRenderer(elements.chartCanvas);
  
  updateUI();
  drawFrame();
}

function resetApp() {
  appState.seedOffset++;
  appState.particles = parseInt(elements.particlesInput.value) || 10;
  appState.amplitude = parseFloat(elements.ampInput.value) || 15.0;
  appState.steps = parseInt(elements.stepsInput.value) || 2000;
  
  const fnConfig = FUNCTIONS[appState.funcId];
  
  seismicEngine.nParticles = appState.particles;
  seismicEngine.noiseAmplitude = appState.amplitude;
  seismicEngine.nSteps = appState.steps;
  
  saEngine.nSteps = appState.steps;
  
  seismicEngine.reset(appState.seedOffset);
  saEngine.reset(appState.seedOffset);
  
  appState.running = false;
  elements.btnPlay.disabled = false;
  elements.btnPause.disabled = true;
  
  if (animationId) cancelAnimationFrame(animationId);
  
  updateUI();
  drawFrame();
}

function updateUI() {
  elements.lblStep.textContent = `${seismicEngine.step}/${seismicEngine.nSteps}`;
  elements.lblBest.textContent = seismicEngine.bestVal.toFixed(4);
  elements.lblAmp.textContent = seismicEngine.currentAmplitude.toFixed(2);
  elements.lblCycle.textContent = seismicEngine.currentCycle;
  elements.progressFill.style.width = `${seismicEngine.progress * 100}%`;
}

function drawFrame() {
  const glow = Math.abs(seismicEngine.currentAmplitude) * 0.5;
  seismicRenderer.renderFrame(
    seismicEngine.particles,
    seismicEngine.trails,
    rffField,
    seismicEngine.t,
    seismicEngine.currentAmplitude,
    appState.showNoise,
    appState.showTrails,
    appState.showOptimum,
    '#00ff88',
    glow
  );
  
  saRenderer.renderFrame(
    [saEngine.pos],
    [saEngine.trail],
    null, 0, 0, false,
    appState.showTrails,
    appState.showOptimum,
    '#ff6b6b',
    0
  );
  
  chartRenderer.render(seismicEngine.bestHistory, saEngine.bestHistory, seismicEngine.nSteps);
}

function tick() {
  let active = false;
  for (let i = 0; i < appState.speed; i++) {
    const s1 = seismicEngine.tick();
    const s2 = saEngine.tick();
    if (s1 || s2) active = true;
  }
  return active;
}

function loop() {
  if (!appState.running) return;
  
  if (tick()) {
    updateUI();
    drawFrame();
    animationId = requestAnimationFrame(loop);
  } else {
    appState.running = false;
    elements.btnPlay.disabled = false;
    elements.btnPause.disabled = true;
  }
}

elements.btnPlay.addEventListener('click', () => {
  if (seismicEngine.step >= seismicEngine.nSteps) resetApp();
  appState.running = true;
  elements.btnPlay.disabled = true;
  elements.btnPause.disabled = false;
  loop();
});

elements.btnPause.addEventListener('click', () => {
  appState.running = false;
  elements.btnPlay.disabled = false;
  elements.btnPause.disabled = true;
});

elements.btnStep.addEventListener('click', () => {
  appState.running = false;
  elements.btnPlay.disabled = false;
  elements.btnPause.disabled = true;
  if (animationId) cancelAnimationFrame(animationId);
  tick();
  updateUI();
  drawFrame();
});

elements.btnReset.addEventListener('click', resetApp);

elements.speedSlider.addEventListener('input', (e) => {
  appState.speed = parseInt(e.target.value);
});

elements.funcSelect.addEventListener('change', (e) => {
  appState.funcId = e.target.value;
  init();
});

elements.ampInput.addEventListener('change', (e) => {
  const amp = parseFloat(e.target.value);
  if (!isNaN(amp)) {
    appState.amplitude = amp;
    seismicEngine.noiseAmplitude = amp;
    updateUI();
  }
});

elements.chkNoise.addEventListener('change', (e) => {
  appState.showNoise = e.target.checked;
  if (!appState.running) drawFrame();
});

elements.chkTrails.addEventListener('change', (e) => {
  appState.showTrails = e.target.checked;
  if (!appState.running) drawFrame();
});

elements.chkOptimum.addEventListener('change', (e) => {
  appState.showOptimum = e.target.checked;
  if (!appState.running) drawFrame();
});

elements.btnPause.disabled = true;
init();