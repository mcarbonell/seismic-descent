// explorer_2d.js - 2D Seismic Explorer Logic with 3D Wireframe Surface Projection

const FUNCTIONS_2D = {
    rastrigin: {
        name: 'Rastrigin',
        range: 5.12,
        globalMin: [0, 0],
        globalMinVal: 0,
        fn: (x, y) => 20 + (x * x - 10 * Math.cos(2 * Math.PI * x)) + (y * y - 10 * Math.cos(2 * Math.PI * y)),
        grad: (x, y) => [
            2 * x + 20 * Math.PI * Math.sin(2 * Math.PI * x),
            2 * y + 20 * Math.PI * Math.sin(2 * Math.PI * y)
        ]
    },
    schwefel: {
        name: 'Schwefel',
        range: 500,
        globalMin: [420.9687, 420.9687],
        globalMinVal: 0,
        fn: (x, y) => 418.9829 * 2 - x * Math.sin(Math.sqrt(Math.abs(x))) - y * Math.sin(Math.sqrt(Math.abs(y))),
        grad: (x, y) => {
            const ax = Math.abs(x) + 1e-9;
            const ay = Math.abs(y) + 1e-9;
            const sx = Math.sqrt(ax);
            const sy = Math.sqrt(ay);
            return [
                -Math.sin(sx) - (sx / 2) * Math.cos(sx) * Math.sign(x),
                -Math.sin(sy) - (sy / 2) * Math.cos(sy) * Math.sign(y)
            ];
        }
    },
    ackley: {
        name: 'Ackley',
        range: 15,
        globalMin: [0, 0],
        globalMinVal: 0,
        fn: (x, y) => {
            const r = Math.sqrt(0.5 * (x * x + y * y)) + 1e-9;
            return -20 * Math.exp(-0.2 * r) - Math.exp(0.5 * (Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y))) + 20 + Math.E;
        },
        grad: (x, y) => {
            const r = Math.sqrt(0.5 * (x * x + y * y)) + 1e-9;
            const term1 = 4 * Math.exp(-0.2 * r) / r;
            const cosSum = 0.5 * (Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y));
            const term2 = Math.PI * Math.exp(cosSum);
            return [
                term1 * 0.5 * x + term2 * Math.sin(2 * Math.PI * x),
                term1 * 0.5 * y + term2 * Math.sin(2 * Math.PI * y)
            ];
        }
    },
    rosenbrock: {
        name: 'Rosenbrock',
        range: 2.0,
        globalMin: [1, 1],
        globalMinVal: 0,
        fn: (x, y) => Math.pow(1 - x, 2) + 100 * Math.pow(y - x * x, 2),
        grad: (x, y) => [
            -2 * (1 - x) - 400 * x * (y - x * x),
            200 * (y - x * x)
        ]
    },
    parabola: {
        name: 'Simple Parabola',
        range: 5.0,
        globalMin: [0, 0],
        globalMinVal: 0,
        fn: (x, y) => 0.5 * (x * x + y * y),
        grad: (x, y) => [x, y]
    }
};

class Explorer2D {
    constructor() {
        this.canvas = document.getElementById('explorerCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // 3D View Angles & Settings
        this.yaw = 0.75;  // Rotation around Z axis (radians)
        this.pitch = 0.70; // Pitch angle (radians)
        this.gridSize = 35; // Grid lines density
        
        this.config = {
            funcKey: 'rastrigin',
            nParticles: 8,
            amplitude: 25, // manual slider value
            dt: 0.005,
            octaves: 4,
            speed: 0.5,
            showBase: true,
            showMorphed: true,
            greedy: true,
            cyclicDt: false,
            autoAmp: false
        };
        
        this.isPlaying = false;
        this.step = 0;
        this.t = 0;
        this.blindAmp = 0.01;
        this.particles = [];
        this.rff = null;
        
        // Mouse dragging logic for rotating the 3D grid
        this.isDragging = false;
        this.lastX = 0;
        this.lastY = 0;
        
        this.initEvents();
        this.resize();
        this.reset();
        
        window.onresize = () => this.resize();
        
        // Start animation loop
        this.loop = this.loop.bind(this);
        requestAnimationFrame(this.loop);
    }
    
    resize() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }
    
    initEvents() {
        // Dragging inputs
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastX = e.clientX;
            this.lastY = e.clientY;
        });
        
        window.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            const dx = e.clientX - this.lastX;
            const dy = e.clientY - this.lastY;
            
            this.yaw += dx * 0.007;
            this.pitch += dy * 0.007;
            
            // Constrain pitch to avoid flipping upside down
            this.pitch = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, this.pitch));
            
            this.lastX = e.clientX;
            this.lastY = e.clientY;
        });
        
        window.addEventListener('mouseup', () => {
            this.isDragging = false;
        });
        
        // UI Bindings
        const bind = (id, configKey, isCheckbox = false, isFloat = false) => {
            const el = document.getElementById(id);
            if (!el) return;
            
            if (isCheckbox) {
                el.onchange = (e) => { this.config[configKey] = e.target.checked; };
                el.checked = this.config[configKey];
            } else {
                el.oninput = (e) => {
                    const val = isFloat ? parseFloat(e.target.value) : parseInt(e.target.value);
                    this.config[configKey] = val;
                    
                    const valLbl = document.getElementById('lbl' + id.charAt(0).toUpperCase() + id.slice(1).replace('Slider', 'Val'));
                    if (valLbl) valLbl.innerText = val;
                    
                    if (configKey === 'nParticles') this.resetParticles();
                    if (configKey === 'octaves') this.reset();
                };
                el.value = this.config[configKey];
            }
        };
        
        bind('speedSlider', 'speed', false, true);
        bind('ampSlider', 'amplitude', false, false);
        bind('dtSlider', 'dt', false, true);
        bind('particlesInput', 'nParticles', false, false);
        bind('octavesInput', 'octaves', false, false);
        bind('chkShowObj', 'showBase', true);
        bind('chkShowMorphed', 'showMorphed', true);
        bind('chkGreedy', 'greedy', true);
        bind('chkCyclicDt', 'cyclicDt', true);
        bind('chkAutoAmp', 'autoAmp', true);
        
        document.getElementById('funcSelect').onchange = (e) => {
            this.config.funcKey = e.target.value;
            this.reset();
        };
        
        document.getElementById('btnPlay').onclick = () => {
            this.isPlaying = true;
            document.getElementById('btnPlay').disabled = true;
            document.getElementById('btnPause').disabled = false;
        };
        document.getElementById('btnPause').onclick = () => {
            this.isPlaying = false;
            document.getElementById('btnPlay').disabled = false;
            document.getElementById('btnPause').disabled = true;
        };
        document.getElementById('btnReset').onclick = () => this.resetParticles();
        document.getElementById('btnPause').disabled = true;
    }
    
    reset() {
        const fnInfo = FUNCTIONS_2D[this.config.funcKey];
        this.rff = new RFFField(64, this.config.octaves, 1, fnInfo.range);
        this.resetParticles();
        this.step = 0;
        this.t = 0;
    }
    
    resetParticles() {
        this.particles = Array.from({ length: this.config.nParticles }, () => ({
            x: Math.random() * 2 - 1, // Normalized [-1, 1] space
            y: Math.random() * 2 - 1,
            history: [] // For drawing trails
        }));
        this.blindAmp = 0.01;
    }
    
    update() {
        const fnInfo = FUNCTIONS_2D[this.config.funcKey];
        const range = fnInfo.range;
        const dtNoise = (10 * Math.PI) / 2000;
        
        let updatesToRun = 0;
        if (this.config.speed >= 1) {
            updatesToRun = Math.floor(this.config.speed);
        } else {
            this.frameCounter = (this.frameCounter || 0) + 1;
            const freq = Math.round(1 / this.config.speed);
            if (this.frameCounter % freq === 0) updatesToRun = 1;
        }
        
        let activeAmplitude = 0;
        for (let s = 0; s < updatesToRun; s++) {
            if (this.config.autoAmp) {
                this.blindAmp *= 1.02;
                activeAmplitude = this.blindAmp;
            } else {
                activeAmplitude = Math.pow(10, (this.config.amplitude / 100) * 5 - 2);
            }
            
            const phaseAmp = activeAmplitude * Math.sin(this.t * 2.0);
            
            let activeDt = this.config.dt;
            if (this.config.cyclicDt) {
                activeDt *= Math.abs(Math.sin(this.t * 1.618));
            }
            
            this.particles.forEach(p => {
                const oldX = p.x;
                const oldY = p.y;
                const realX = p.x * range;
                const realY = p.y * range;
                
                const [gObjX, gObjY] = fnInfo.grad(realX, realY);
                const rffRes = this.rff.noiseAndGrad(realX, realY, this.t, phaseAmp);
                
                // Total Gradients scaled back to normalized coordinates
                let gradNormX = (gObjX + rffRes.gradX) * range;
                let gradNormY = (gObjY + rffRes.gradY) * range;
                
                // Gradient clipping to prevent flying off the screen
                const maxStep = 0.03;
                const stepLen = Math.sqrt(gradNormX * gradNormX + gradNormY * gradNormY) * activeDt;
                if (stepLen > maxStep) {
                    const scale = maxStep / stepLen;
                    gradNormX *= scale;
                    gradNormY *= scale;
                }
                
                const nextX = p.x - activeDt * gradNormX;
                const nextY = p.y - activeDt * gradNormY;
                
                if (this.config.greedy) {
                    const rffNext = this.rff.noiseAndGrad(nextX * range, nextY * range, this.t, phaseAmp);
                    const valOld = fnInfo.fn(realX, realY) + rffRes.noise;
                    const valNext = fnInfo.fn(nextX * range, nextY * range) + rffNext.noise;
                    if (valNext < valOld) {
                        p.x = nextX;
                        p.y = nextY;
                    }
                } else {
                    p.x = nextX;
                    p.y = nextY;
                }
                
                // Clamp within bounds
                p.x = Math.max(-1, Math.min(1, p.x));
                p.y = Math.max(-1, Math.min(1, p.y));
                
                p.history.push({ x: p.x, y: p.y });
                if (p.history.length > 50) p.history.shift();
            });
            
            this.t += dtNoise;
            this.step++;
        }
        
        // Calculate Success Rate (SR)
        const optX = fnInfo.globalMin[0] / range;
        const optY = fnInfo.globalMin[1] / range;
        const eps = 0.05;
        const solved = this.particles.filter(p => {
            const dist = Math.sqrt(Math.pow(p.x - optX, 2) + Math.pow(p.y - optY, 2));
            return dist < eps;
        });
        const successRate = ((solved.length / this.particles.length) * 100).toFixed(0);
        
        // Update stats
        document.getElementById('lblStep').innerText = this.step;
        
        let displayAmp = activeAmplitude * Math.sin(this.t * 2.0);
        document.getElementById('lblActiveAmp').innerText = Math.abs(displayAmp) > 1000 ? displayAmp.toExponential(1) : displayAmp.toFixed(2);
        
        const avgVal = this.particles.reduce((acc, p) => acc + fnInfo.fn(p.x * range, p.y * range), 0) / this.particles.length;
        document.getElementById('lblAvgVal').innerText = avgVal.toFixed(4);
        
        document.getElementById('lblSuccess').innerText = successRate + '%';
        document.getElementById('progressFill').style.width = ((this.step % 2000) / 20) + '%';
    }
    
    project3D(u, v, z) {
        // Rotations
        const cosY = Math.cos(this.yaw);
        const sinY = Math.sin(this.yaw);
        const cosP = Math.cos(this.pitch);
        const sinP = Math.sin(this.pitch);
        
        // Rotate around Z axis (Yaw)
        const x1 = u * cosY - v * sinY;
        const y1 = u * sinY + v * cosY;
        const z1 = z;
        
        // Rotate around X axis (Pitch)
        const x2 = x1;
        const y2 = y1 * cosP - z1 * sinP;
        const z2 = y1 * sinP + z1 * cosP;
        
        // Camera Projection
        const dist = 3.5;
        const zoom = 240;
        
        const cx = this.canvas.width / 2;
        const cy = this.canvas.height / 2 + 30; // slide slightly down
        
        const denom = z2 + dist;
        const px = cx + (zoom * x2) / denom;
        const py = cy + (zoom * y2) / denom;
        
        return { px, py, depth: z2 };
    }
    
    render() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        const fnInfo = FUNCTIONS_2D[this.config.funcKey];
        const range = fnInfo.range;
        
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(0, 0, w, h);
        
        // Determine active amplitude
        let activeAmplitude = 0;
        if (this.config.autoAmp) {
            activeAmplitude = this.blindAmp;
        } else {
            activeAmplitude = Math.pow(10, (this.config.amplitude / 100) * 5 - 2);
        }
        const phaseAmp = activeAmplitude * Math.sin(this.t * 2.0);
        
        // 1. Grid Precomputation
        const N = this.gridSize;
        const projectedGrid = [];
        
        // Pre-sample grid heights to normalize Z scale dynamically
        let zMinVal = Infinity;
        let zMaxVal = -Infinity;
        const rawZGrid = new Float32Array((N + 1) * (N + 1));
        const rawZNoise = new Float32Array((N + 1) * (N + 1));
        
        for (let j = 0; j <= N; j++) {
            const v = -1 + 2 * (j / N);
            const y = v * range;
            for (let i = 0; i <= N; i++) {
                const u = -1 + 2 * (i / N);
                const x = u * range;
                
                const baseZ = fnInfo.fn(x, y);
                const noiseVal = this.rff.noiseAndGrad(x, y, this.t, phaseAmp).noise;
                
                const idx = j * (N + 1) + i;
                rawZGrid[idx] = baseZ;
                rawZNoise[idx] = noiseVal;
                
                const totalZ = baseZ + noiseVal;
                zMinVal = Math.min(zMinVal, totalZ);
                zMaxVal = Math.max(zMaxVal, totalZ);
            }
        }
        
        // Smooth Z scale factor (fits Z nicely inside [-0.7, 0.7])
        const zRange = (zMaxVal - zMinVal) || 1.0;
        const scaleZ = (val) => ((val - zMinVal) / zRange - 0.5) * 1.3;
        
        // Project all grid points
        for (let j = 0; j <= N; j++) {
            const v = -1 + 2 * (j / N);
            projectedGrid.push([]);
            for (let i = 0; i <= N; i++) {
                const u = -1 + 2 * (i / N);
                const idx = j * (N + 1) + i;
                
                const zMorphedNorm = scaleZ(rawZGrid[idx] + rawZNoise[idx]);
                const zBaseNorm = scaleZ(rawZGrid[idx]);
                
                const ptMorphed = this.project3D(u, v, zMorphedNorm);
                const ptBase = this.project3D(u, v, zBaseNorm);
                
                projectedGrid[j].push({
                    morph: ptMorphed,
                    base: ptBase,
                    rawVal: rawZGrid[idx] + rawZNoise[idx],
                    normVal: (rawZGrid[idx] + rawZNoise[idx] - zMinVal) / zRange
                });
            }
        }
        
        // 2. Draw Base (Static) Grid
        if (this.config.showBase) {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
            ctx.lineWidth = 1;
            
            for (let j = 0; j <= N; j++) {
                // Draw horizontal base lines
                ctx.beginPath();
                for (let i = 0; i <= N; i++) {
                    const pt = projectedGrid[j][i].base;
                    if (i === 0) ctx.moveTo(pt.px, pt.py);
                    else ctx.lineTo(pt.px, pt.py);
                }
                ctx.stroke();
            }
            for (let i = 0; i <= N; i++) {
                // Draw vertical base lines
                ctx.beginPath();
                for (let j = 0; j <= N; j++) {
                    const pt = projectedGrid[j][i].base;
                    if (j === 0) ctx.moveTo(pt.px, pt.py);
                    else ctx.lineTo(pt.px, pt.py);
                }
                ctx.stroke();
            }
        }
        
        // 3. Draw Morphed Grid
        if (this.config.showMorphed) {
            ctx.lineWidth = 1;
            
            // To make lines fade in the distance, we draw them with depth-based alphas
            for (let j = 0; j <= N; j++) {
                for (let i = 0; i <= N; i++) {
                    const pt = projectedGrid[j][i].morph;
                    const depthAlpha = Math.max(0.1, Math.min(1.0, 1.2 - (pt.depth + 1.0) / 2.0));
                    
                    // Draw horizontal connections
                    if (i < N) {
                        const nextPt = projectedGrid[j][i+1].morph;
                        const avgVal = (projectedGrid[j][i].normVal + projectedGrid[j][i+1].normVal) / 2;
                        
                        // HSL colors mapping: low is cyan/green (hue 140), high is violet/purple (hue 280)
                        const hue = 140 + avgVal * 140;
                        ctx.strokeStyle = `hsla(${hue}, 80%, 50%, ${depthAlpha * 0.45})`;
                        
                        ctx.beginPath();
                        ctx.moveTo(pt.px, pt.py);
                        ctx.lineTo(nextPt.px, nextPt.py);
                        ctx.stroke();
                    }
                    
                    // Draw vertical connections
                    if (j < N) {
                        const nextPt = projectedGrid[j+1][i].morph;
                        const avgVal = (projectedGrid[j][i].normVal + projectedGrid[j+1][i].normVal) / 2;
                        const hue = 140 + avgVal * 140;
                        ctx.strokeStyle = `hsla(${hue}, 80%, 50%, ${depthAlpha * 0.45})`;
                        
                        ctx.beginPath();
                        ctx.moveTo(pt.px, pt.py);
                        ctx.lineTo(nextPt.px, nextPt.py);
                        ctx.stroke();
                    }
                }
            }
        }
        
        // 4. Draw Optimum marker projected onto morphed terrain
        const realOptX = fnInfo.globalMin[0];
        const realOptY = fnInfo.globalMin[1];
        const optZ = fnInfo.fn(realOptX, realOptY) + this.rff.noiseAndGrad(realOptX, realOptY, this.t, phaseAmp).noise;
        const optProj = this.project3D(realOptX / range, realOptY / range, scaleZ(optZ));
        
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(optProj.px, optProj.py, 6, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(optProj.px - 9, optProj.py);
        ctx.lineTo(optProj.px + 9, optProj.py);
        ctx.moveTo(optProj.px, optProj.py - 9);
        ctx.lineTo(optProj.px, optProj.py + 9);
        ctx.stroke();
        
        // 5. Draw Trails of Particles
        this.particles.forEach(p => {
            if (p.history.length < 2) return;
            
            ctx.lineWidth = 2.5;
            for (let k = 1; k < p.history.length; k++) {
                const ptOld = p.history[k-1];
                const ptNew = p.history[k];
                
                const realXOld = ptOld.x * range;
                const realYOld = ptOld.y * range;
                const zOld = fnInfo.fn(realXOld, realYOld) + this.rff.noiseAndGrad(realXOld, realYOld, this.t, phaseAmp).noise;
                const projOld = this.project3D(ptOld.x, ptOld.y, scaleZ(zOld));
                
                const realXNew = ptNew.x * range;
                const realYNew = ptNew.y * range;
                const zNew = fnInfo.fn(realXNew, realYNew) + this.rff.noiseAndGrad(realXNew, realYNew, this.t, phaseAmp).noise;
                const projNew = this.project3D(ptNew.x, ptNew.y, scaleZ(zNew));
                
                const ageAlpha = k / p.history.length;
                ctx.strokeStyle = `rgba(0, 255, 136, ${ageAlpha * 0.4})`;
                
                ctx.beginPath();
                ctx.moveTo(projOld.px, projOld.py);
                ctx.lineTo(projNew.px, projNew.py);
                ctx.stroke();
            }
        });
        
        // 6. Draw Particles
        this.particles.forEach(p => {
            const realX = p.x * range;
            const realY = p.y * range;
            const z = fnInfo.fn(realX, realY) + this.rff.noiseAndGrad(realX, realY, this.t, phaseAmp).noise;
            const proj = this.project3D(p.x, p.y, scaleZ(z));
            
            // Adjust sphere radius by camera depth
            const baseRad = 6;
            const dist = 3.5;
            const radius = baseRad * (dist / (proj.depth + dist));
            
            // Render particle as bright green sphere
            ctx.beginPath();
            ctx.arc(proj.px, proj.py, Math.max(1, radius), 0, 2 * Math.PI);
            ctx.fillStyle = '#00ff88';
            ctx.shadowBlur = 12;
            ctx.shadowColor = '#00ff88';
            ctx.fill();
            ctx.shadowBlur = 0; // reset
            
            // Sphere shading highlight
            ctx.beginPath();
            ctx.arc(proj.px - radius * 0.3, proj.py - radius * 0.3, radius * 0.2, 0, 2 * Math.PI);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
        });
        
        // Camera rotation info
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.font = '10px JetBrains Mono';
        ctx.textAlign = 'left';
        ctx.fillText(`Yaw: ${(this.yaw * 180 / Math.PI).toFixed(0)}° | Pitch: ${(this.pitch * 180 / Math.PI).toFixed(0)}°`, 15, h - 15);
    }
    
    loop() {
        if (this.isPlaying) {
            this.update();
        }
        this.render();
        requestAnimationFrame(this.loop);
    }
}

// Instantiate the Explorer once DOM is loaded
window.addEventListener('load', () => {
    window.explorer = new Explorer2D();
});
