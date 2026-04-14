const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d');
const offscreenCanvas = document.createElement('canvas');
const offscreenCtx = offscreenCanvas.getContext('2d');
const RES = 100; // Resolution for the noise heatmap (upscaled via canvas)
offscreenCanvas.width = RES;
offscreenCanvas.height = RES;

let morphSteps = 100;
let dt = 0.02;
let octaves = 4;
let running = false;
let animationId;

let seedA = 1;
let seedB = 2;
// The domain is [0, 1]. RFF Field expects searchRange to scale frequencies.
// Setting searchRange to 0.5 means [-0.5, 0.5], which is exactly size 1.0.
let fieldA = new RFFField(64, octaves, seedA, 0.5);
let fieldB = new RFFField(64, octaves, seedB, 0.5);

let particle = { x: 0.5, y: 0.5 };
let trail = [];

const GRID_SIZE = 40;
let visited = new Uint8Array(GRID_SIZE * GRID_SIZE);
let visitedCount = 0;
let step = 0;

function reset() {
    seedA = 1;
    seedB = 2;
    step = 0;
    fieldA = new RFFField(64, octaves, seedA, 0.5);
    fieldB = new RFFField(64, octaves, seedB, 0.5);
    particle = { x: 0.5, y: 0.5 };
    trail = [];
    visited.fill(0);
    visitedCount = 0;
    draw();
}

function tick() {
    let u = (step % morphSteps) / morphSteps;
    
    // Constant variance interpolation for independent Gaussians
    // cos^2 + sin^2 = 1
    const weightA = Math.cos(u * Math.PI / 2);
    const weightB = Math.sin(u * Math.PI / 2);
    
    // Shift particle to [-0.5, 0.5] for RFF evaluation
    let px = particle.x - 0.5;
    let py = particle.y - 0.5;
    
    let resA = fieldA.noiseAndGrad(px, py, 0, 1.0);
    let resB = fieldB.noiseAndGrad(px, py, 0, 1.0);
    
    let gradX = weightA * resA.gradX + weightB * resB.gradX;
    let gradY = weightA * resA.gradY + weightB * resB.gradY;
    
    // Move particle "downhill" in the noise landscape
    particle.x -= dt * gradX;
    particle.y -= dt * gradY;
    
    // Soft bounce / clip at the edges of the [0, 1] domain
    if (particle.x < 0) { particle.x = 0; }
    if (particle.x > 1) { particle.x = 1; }
    if (particle.y < 0) { particle.y = 0; }
    if (particle.y > 1) { particle.y = 1; }
    
    trail.push({x: particle.x, y: particle.y});
    if (trail.length > 150) trail.shift();
    
    // Update coverage grid
    let gx = Math.floor(particle.x * GRID_SIZE);
    let gy = Math.floor(particle.y * GRID_SIZE);
    gx = Math.max(0, Math.min(GRID_SIZE - 1, gx));
    gy = Math.max(0, Math.min(GRID_SIZE - 1, gy));
    let idx = gy * GRID_SIZE + gx;
    
    if (visited[idx] === 0) {
        visited[idx] = 1;
        visitedCount++;
    }
    
    step++;
    
    // When we finish morphing to B, B becomes A, and we generate a new B
    if (step % morphSteps === 0) {
        seedA = seedB;
        seedB++;
        fieldA = fieldB;
        fieldB = new RFFField(64, octaves, seedB, 0.5);
    }
}

function draw() {
    const imageData = offscreenCtx.createImageData(RES, RES);
    let u = (step % morphSteps) / morphSteps;
    const weightA = Math.cos(u * Math.PI / 2);
    const weightB = Math.sin(u * Math.PI / 2);
    
    let maxAmp = 1.5; // Heuristic for normalization
    
    for (let py = 0; py < RES; py++) {
        for (let px = 0; px < RES; px++) {
            let x = (px / RES) - 0.5;
            let y = (py / RES) - 0.5;
            
            let nA = fieldA.noiseAndGrad(x, y, 0, 1.0).noise;
            let nB = fieldB.noiseAndGrad(x, y, 0, 1.0).noise;
            let n = weightA * nA + weightB * nB;
            
            let t = (n + maxAmp) / (2 * maxAmp);
            const [r, g, b] = viridisColor(t); // function from renderer.js
            
            let idx = (py * RES + px) * 4;
            imageData.data[idx] = r;
            imageData.data[idx+1] = g;
            imageData.data[idx+2] = b;
            imageData.data[idx+3] = 255;
        }
    }
    
    offscreenCtx.putImageData(imageData, 0, 0);
    ctx.drawImage(offscreenCanvas, 0, 0, canvas.width, canvas.height);
    
    // Draw coverage overlay
    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
    let cellW = canvas.width / GRID_SIZE;
    for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        if (visited[i]) {
            let gx = i % GRID_SIZE;
            let gy = Math.floor(i / GRID_SIZE);
            ctx.fillRect(gx * cellW, gy * cellW, cellW, cellW);
        }
    }
    
    // Draw trail
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.8)';
    ctx.lineWidth = 2;
    if (trail.length > 0) {
        ctx.moveTo(trail[0].x * canvas.width, trail[0].y * canvas.height);
        for(let i=1; i<trail.length; i++) {
            ctx.lineTo(trail[i].x * canvas.width, trail[i].y * canvas.height);
        }
        ctx.stroke();
    }
    
    // Draw particle
    ctx.beginPath();
    ctx.fillStyle = '#ff0044'; // Red particle
    ctx.arc(particle.x * canvas.width, particle.y * canvas.height, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Update UI labels
    document.getElementById('lblCoverage').textContent = ((visitedCount / (GRID_SIZE * GRID_SIZE)) * 100).toFixed(1) + '%';
    document.getElementById('lblSeedA').textContent = seedA;
    document.getElementById('lblSeedB').textContent = seedB;
    document.getElementById('lblMorph').textContent = Math.floor(u * 100) + '%';
    document.getElementById('lblStepCount').textContent = step;
}

function loop() {
    if (!running) return;
    tick();
    draw();
    animationId = requestAnimationFrame(loop);
}

// UI Event Listeners
document.getElementById('btnPlay').addEventListener('click', () => {
    running = true;
    document.getElementById('btnPlay').style.background = 'var(--bg-color)';
    document.getElementById('btnPlay').style.color = 'var(--text-color)';
    document.getElementById('btnPlay').disabled = true;
    document.getElementById('btnPause').disabled = false;
    loop();
});

document.getElementById('btnPause').addEventListener('click', () => {
    running = false;
    document.getElementById('btnPlay').style.background = 'var(--seismic-color)';
    document.getElementById('btnPlay').style.color = '#000';
    document.getElementById('btnPlay').disabled = false;
    document.getElementById('btnPause').disabled = true;
});

document.getElementById('btnReset').addEventListener('click', reset);

document.getElementById('morphStepsSlider').addEventListener('input', (e) => {
    morphSteps = parseInt(e.target.value);
    document.getElementById('lblMorphSteps').textContent = morphSteps;
});

document.getElementById('dtSlider').addEventListener('input', (e) => {
    dt = parseFloat(e.target.value);
    document.getElementById('lblDt').textContent = dt.toFixed(3);
});

document.getElementById('octavesSlider').addEventListener('input', (e) => {
    octaves = parseInt(e.target.value);
    document.getElementById('lblOctaves').textContent = octaves;
    fieldA = new RFFField(64, octaves, seedA, 0.5);
    fieldB = new RFFField(64, octaves, seedB, 0.5);
    if (!running) draw();
});

// Initial draw
reset();