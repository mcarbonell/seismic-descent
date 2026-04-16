// explorer_1d.js - 1D Seismic Explorer Logic

const FUNCTIONS = {
    rastrigin: {
        name: 'Rastrigin',
        range: 5.12,
        globalMin: 0,
        globalMinVal: 0,
        fn: (x) => 10 + (x * x - 10 * Math.cos(2 * Math.PI * x)),
        grad: (x) => 2 * x + 20 * Math.PI * Math.sin(2 * Math.PI * x)
    },
    schwefel: {
        name: 'Schwefel',
        range: 500,
        globalMin: 420.9687,
        globalMinVal: 0,
        fn: (x) => 418.9829 - x * Math.sin(Math.sqrt(Math.abs(x))),
        grad: (x) => {
            const ax = Math.abs(x) + 1e-9;
            const sx = Math.sqrt(ax);
            return -Math.sin(sx) - (sx / 2) * Math.cos(sx);
        }
    },
    ackley: {
        name: 'Ackley',
        range: 15,
        globalMin: 0,
        globalMinVal: 0,
        fn: (x) => {
            const a = 20, b = 0.2, c = 2 * Math.PI;
            return -a * Math.exp(-b * Math.abs(x)) - Math.exp(Math.cos(c * x)) + a + Math.E;
        },
        grad: (x) => {
            const a = 20, b = 0.2, c = 2 * Math.PI;
            const ax = Math.abs(x) + 1e-9;
            const term1 = a * b * Math.exp(-b * ax) * (x / ax);
            const term2 = Math.exp(Math.cos(c * x)) * c * Math.sin(c * x);
            return term1 + term2;
        }
    },
    parabola: {
        name: 'Simple Parabola',
        range: 5,
        globalMin: 0,
        globalMinVal: 0,
        fn: (x) => 0.5 * x * x,
        grad: (x) => x
    },
    griewank: {
        name: 'Griewank',
        range: 100,
        globalMin: 0,
        globalMinVal: 0,
        fn: (x) => (x * x) / 4000 - Math.cos(x) + 1,
        grad: (x) => x / 2000 + Math.sin(x)
    },
    rosenbrock: {
        name: 'Rosenbrock (Slice)',
        range: 2,
        globalMin: 1,
        globalMinVal: 0,
        fn: (x) => Math.pow(1 - x, 2) + 100 * Math.pow(1 - x * x, 2),
        grad: (x) => 400 * Math.pow(x, 3) - 398 * x - 2
    },
    levy: {
        name: 'Levy',
        range: 10,
        globalMin: 1,
        globalMinVal: 0,
        fn: (x) => {
            const w = 1 + (x - 1) / 4;
            return Math.pow(Math.sin(Math.PI * w), 2) + Math.pow(w - 1, 2) * (1 + Math.pow(Math.sin(2 * Math.PI * w), 2));
        },
        grad: (x) => {
            const w = 1 + (x - 1) / 4;
            const term1 = Math.PI * Math.sin(2 * Math.PI * w);
            const term2 = 2 * (w - 1) * (1 + Math.pow(Math.sin(2 * Math.PI * w), 2));
            const term3 = Math.pow(w - 1, 2) * 2 * Math.PI * Math.sin(4 * Math.PI * w);
            return (term1 + term2 + term3) / 4;
        }
    },
    michalewicz: {
        name: 'Michalewicz',
        range: Math.PI,
        globalMin: 2.2, // Approx for 1D
        globalMinVal: -1,
        fn: (x) => -Math.sin(x) * Math.pow(Math.sin(x * x / Math.PI), 20),
        grad: (x) => {
            const s1 = Math.sin(x);
            const c1 = Math.cos(x);
            const s2 = Math.sin(x * x / Math.PI);
            const c2 = Math.cos(x * x / Math.PI);
            const p = Math.pow(s2, 19);
            return -(c1 * s2 * p + s1 * 20 * p * c2 * (2 * x / Math.PI));
        }
    },
    styblinski_tang: {
        name: 'Styblinski-Tang',
        range: 5,
        globalMin: -2.903534,
        globalMinVal: -39.16616 / 2, // 1D value
        fn: (x) => 0.5 * (Math.pow(x, 4) - 16 * x * x + 5 * x),
        grad: (x) => 0.5 * (4 * Math.pow(x, 3) - 32 * x + 5)
    },
    gramacy_lee: {
        name: 'Gramacy & Lee',
        range: 1.5, // Center around 1.5 with range 1.5 -> [0, 3] approx
        offset: 1.5, 
        globalMin: 0.5485,
        globalMinVal: -0.869,
        fn: (x) => {
            // Adjust x to be in [0.5, 2.5] based on our [-1, 1] normalization
            // In explorer_1d, realX = p.x * range. 
            // If range is 1.5 and we add an offset of 1.5:
            const val = x + 0.5; // Shift to start from 0.5
            if (val <= 0) return 0;
            return Math.sin(10 * Math.PI * val) / (2 * val) + Math.pow(val - 1, 4);
        },
        grad: (x) => {
            const val = x + 0.5;
            if (val <= 0) return 0;
            const term1 = (10 * Math.PI * Math.cos(10 * Math.PI * val)) / (2 * val);
            const term2 = Math.sin(10 * Math.PI * val) / (2 * val * val);
            const term3 = 4 * Math.pow(val - 1, 3);
            return term1 - term2 + term3;
        }
    }
};

class Explorer1D {
    constructor() {
        this.canvas = document.getElementById('explorerCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.particles = [];
        this.heatmap = null;
        this.heatmapRes = 500;
        
        this.rff = null;
        this.t = 0;
        this.step = 0;
        this.isPlaying = false;
        
        this.config = {
            funcKey: 'rastrigin',
            nParticles: 1,
            amplitude: 15,
            dt: 0.002,
            octaves: 4,
            speed: 0.5, // Slower by default
            showObj: true,
            showNoise: true,
            showTotal: true,
            showHeatmap: true,
            greedy: true,
            cyclicDt: false,
            autoAmp: false
        };

        this.frameCounter = 0;

        this.initUI();
        this.reset();
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    initUI() {
        const bind = (id, key, event = 'input', cb = null) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.addEventListener(event, (e) => {
                let val;
                if (el.type === 'checkbox') {
                    val = el.checked;
                } else if (el.type === 'number' || el.type === 'range') {
                    val = parseFloat(el.value);
                } else {
                    val = el.value;
                }
                this.config[key] = val;
                if (cb) cb(val);
            });
        };

        bind('funcSelect', 'funcKey', 'change', () => this.reset());
        bind('particlesInput', 'nParticles', 'change', () => this.reset());
        bind('ampSlider', 'amplitude', 'input', (v) => document.getElementById('lblAmpVal').innerText = v);
        bind('dtSlider', 'dt', 'input', (v) => document.getElementById('lblDtVal').innerText = v);
        bind('octavesInput', 'octaves', 'change', () => this.reset()); // Re-init RFF when octaves change
        bind('speedSlider', 'speed', 'input');
        bind('chkShowObj', 'showObj');
        bind('chkShowNoise', 'showNoise');
        bind('chkShowTotal', 'showTotal');
        bind('chkShowHeatmap', 'showHeatmap');
        bind('chkGreedy', 'greedy');
        bind('chkCyclicDt', 'cyclicDt');
        bind('chkAutoAmp', 'autoAmp');

        document.getElementById('btnPlay').onclick = () => this.isPlaying = true;
        document.getElementById('btnPause').onclick = () => this.isPlaying = false;
        document.getElementById('btnReset').onclick = () => this.resetParticles();
        document.getElementById('btnClearHeatmap').onclick = () => this.clearHeatmap();
    }

    resize() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }

    reset() {
        this.rff = new RFFField(64, this.config.octaves, 1, FUNCTIONS[this.config.funcKey].range);
        this.clearHeatmap();
        this.resetParticles();
        this.step = 0;
        this.t = 0;
    }

    resetParticles() {
        // Now working in normalized [-1, 1] space
        this.particles = Array.from({ length: this.config.nParticles }, () => ({
            x: Math.random() * 2 - 1,
            val: 0,
            history: [] // For distance-based stagnation
        }));
        this.globalBest = Infinity;
        this.blindAmp = 0.01; // Starts blindly small
    }

    clearHeatmap() {
        this.heatmap = new Float32Array(this.heatmapRes).fill(0);
    }

    update() {
        const fnInfo = FUNCTIONS[this.config.funcKey];
        const range = fnInfo.range;
        const dtNoise = (10 * Math.PI) / 2000; 
        
        let updatesToRun = 0;
        if (this.config.speed >= 1) {
            updatesToRun = Math.floor(this.config.speed);
        } else {
            this.frameCounter++;
            const freq = Math.round(1 / this.config.speed);
            if (this.frameCounter % freq === 0) updatesToRun = 1;
        }

        let currentFrameActiveAmp = 0;

        for (let s = 0; s < updatesToRun; s++) {
            let activeAmplitude = 0;
            if (this.config.autoAmp) {
                // Pure BBO: Grow exponentially. 2% growth per stagnant step
                this.blindAmp *= 1.02;
                activeAmplitude = this.blindAmp;
            } else {
                // Manual slider: map 0-100 to an absolute log scale (10^-2 to 10^3)
                activeAmplitude = Math.pow(10, (this.config.amplitude / 100) * 5 - 2);
            }
            currentFrameActiveAmp = activeAmplitude;

            const phaseAmp = activeAmplitude * Math.sin(this.t * 2.0);
            
            // Dynamic dt calculation (decoupled frequency from noise amplitude)
            let activeDt = this.config.dt;
            if (this.config.cyclicDt) {
                activeDt *= Math.abs(Math.sin(this.t * 1.618)); // Golden ratio for non-resonance
            }
            
            this.particles.forEach(p => {
                // p.x is in [-1, 1]
                const oldX = p.x;
                const realX = p.x * range;
                
                // Real Gradients
                const gObj = fnInfo.grad(realX);
                // We use the real range for RFF to keep frequency meaning consistent
                const rffRes = this.rff.noiseAndGrad(realX, 0, this.t, phaseAmp);
                const gNoise = rffRes.gradX;
                
                // Normalized Gradient: g_norm = g_real * range
                let totalGradNorm = (gObj + gNoise) * range;
                
                // Clipping in normalized space
                const maxStep = 0.02; // Max 1% of domain (size 2) per step
                if (Math.abs(activeDt * totalGradNorm) > maxStep) {
                    totalGradNorm = (Math.sign(totalGradNorm) * maxStep) / activeDt;
                }
                
                const nextX = p.x - activeDt * totalGradNorm;
                
                if (this.config.greedy) {
                    // Check on real scale
                    const rffNext = this.rff.noiseAndGrad(nextX * range, 0, this.t, phaseAmp);
                    const valOld = fnInfo.fn(realX) + rffRes.noise;
                    const valNext = fnInfo.fn(nextX * range) + rffNext.noise;
                    if (valNext < valOld) p.x = nextX;
                } else {
                    p.x = nextX;
                }
                
                if (p.x < -1) p.x = -1;
                if (p.x > 1) p.x = 1;
                
                // Continuous Heatmap Accumulation (no gaps)
                const startX = Math.min(oldX, p.x);
                const endX = Math.max(oldX, p.x);
                const startIdx = Math.max(0, Math.floor(((startX + 1) / 2) * this.heatmapRes));
                const endIdx = Math.min(this.heatmapRes - 1, Math.floor(((endX + 1) / 2) * this.heatmapRes));
                
                if (startIdx === endIdx) {
                    this.heatmap[startIdx] += 0.1;
                } else {
                    const span = endIdx - startIdx + 1;
                    const val = 0.1 / span; // Distribute the visit weight
                    for (let i = startIdx; i <= endIdx; i++) {
                        this.heatmap[i] += val;
                    }
                }
                
                // Track history for Auto-Amp
                p.history.push(p.x);
                if (p.history.length > 50) p.history.shift();
            });

            // Distance-based Stagnation (Spatial Spread) evaluated discretely every 50 steps
            if (this.config.autoAmp && this.step > 0 && this.step % 50 === 0) {
                let maxSpread = 0;
                this.particles.forEach(p => {
                    let minX = Infinity;
                    let maxX = -Infinity;
                    for (let i = 0; i < p.history.length; i++) {
                        if (p.history[i] < minX) minX = p.history[i];
                        if (p.history[i] > maxX) maxX = p.history[i];
                    }
                    const spread = maxX - minX;
                    if (spread > maxSpread) maxSpread = spread;
                });

                // Discretized Control Loop Actions
                if (maxSpread < 0.04) {
                    this.blindAmp *= 2.0; // Double the amplitude if stuck after full window
                } else if (maxSpread > 0.08) {
                    this.blindAmp *= 0.5; // Halve it if moving quickly
                }
                
                // Limits to prevent floating point absurdities
                if (this.blindAmp < 0.01) this.blindAmp = 0.01; 
                if (this.blindAmp > 1e6) this.blindAmp = 1e6; // Absolute mathematical ceiling
            }

            this.t += dtNoise;
            this.step++;
        }
        
        const normEpsilon = 0.01; // 1% of normalized domain
        const normMin = fnInfo.globalMin / range;
        const successfulOnes = this.particles.filter(p => Math.abs(p.x - normMin) < normEpsilon);
        const successRate = (successfulOnes.length / this.particles.length * 100).toFixed(0);

        // Update stats
        document.getElementById('lblStep').innerText = this.step;
        
        let displayAmp = currentFrameActiveAmp * Math.sin(this.t * 2.0);
        document.getElementById('lblActiveAmp').innerText = Math.abs(displayAmp) > 1000 ? displayAmp.toExponential(1) : displayAmp.toFixed(2);
        
        const avgDiv = this.particles.reduce((a, b) => a + fnInfo.fn(b.x * range), 0) / this.particles.length;
        document.getElementById('lblAvgVal').innerText = avgDiv.toFixed(4);
        
        const coverage = (this.heatmap.filter(v => v > 0).length / this.heatmapRes * 100).toFixed(1);
        document.getElementById('lblCoverage').innerText = coverage + '% (SR: ' + successRate + '%)';
        document.getElementById('progressFill').style.width = (this.step % 2000 / 20) + '%';
    }

    render() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        const fnInfo = FUNCTIONS[this.config.funcKey];
        const range = fnInfo.range;
        
        ctx.clearRect(0, 0, w, h);

        // Pre-calculate rendering amplitude for dynamic camera bounds
        let renderAmplitude;
        if (this.config.autoAmp) {
            renderAmplitude = this.blindAmp;
        } else {
            renderAmplitude = Math.pow(10, (this.config.amplitude / 100) * 5 - 2);
        }
        const amp = renderAmplitude * Math.sin(this.t * 2.0);
        
        // Sampling for Y scale based on original function
        let yMinObj = Infinity, yMaxObj = -Infinity;
        const samples = 100;
        for (let i = 0; i <= samples; i++) {
            const u = -1 + (i / samples) * 2;
            const x = u * range;
            const yObj = fnInfo.fn(x);
            yMinObj = Math.min(yMinObj, yObj);
            yMaxObj = Math.max(yMaxObj, yObj);
        }
        
        // Dynamic camera bounds: stretch Y view when amplitude explodes
        const yMin = yMinObj - renderAmplitude * 1.5;
        const yMax = yMaxObj + renderAmplitude * 1.5;
        
        // Add some margin to Y
        const yRange = yMax - yMin;
        const pad = Math.max(yRange * 0.1, 0.1); 
        const viewYMin = yMin - pad;
        const viewYMax = yMax + pad;
        
        const toX = (u) => ((u + 1) / 2) * w;
        const toY = (y) => h - ((y - viewYMin) / (viewYMax - viewYMin)) * h;

        // Draw Heatmap
        if (this.config.showHeatmap) {
            const barW = w / this.heatmapRes;
            const maxVal = Math.max(...this.heatmap, 0.1);
            ctx.fillStyle = 'rgba(0, 255, 136, 0.1)';
            for (let i = 0; i < this.heatmapRes; i++) {
                if (this.heatmap[i] > 0) {
                    const barH = (Math.min(this.heatmap[i], maxVal) / maxVal) * h;
                    ctx.fillRect(i * barW, h - barH, barW, barH);
                }
            }
        }

        // Draw Axis Labels / Ranges
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.font = '10px JetBrains Mono';
        ctx.textAlign = 'left';
        ctx.fillText(`Y max: ${viewYMax.toFixed(1)}`, 10, 20);
        ctx.fillText(`Y min: ${viewYMin.toFixed(1)}`, 10, h - 10);
        
        ctx.textAlign = 'center';
        ctx.fillText(`X: -${range}`, 20, h / 2);
        ctx.fillText(`X: ${range}`, w - 20, h / 2);
        
        ctx.fillStyle = 'rgba(136, 136, 160, 0.8)';
        ctx.fillText(`Domain: [${-range}, ${range}] | View Range: [${viewYMin.toFixed(0)}, ${viewYMax.toFixed(0)}]`, w/2, 20);

        
        // Draw Curves
        const drawCurve = (color, fn_norm, alpha = 0.5) => {
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.globalAlpha = alpha;
            ctx.lineWidth = 2;
            for (let i = 0; i <= w; i += 2) {
                const u = (i / w) * 2 - 1;
                const yVal = fn_norm(u);
                if (i === 0) ctx.moveTo(i, toY(yVal));
                else ctx.lineTo(i, toY(yVal));
            }
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        };

        if (this.config.showObj) {
            drawCurve('#555', (u) => fnInfo.fn(u * range), 0.8);
        }
        
        if (this.config.showNoise) {
            drawCurve('#ff6b6b', (u) => {
                const rffRes = this.rff.noiseAndGrad(u * range, 0, this.t, amp);
                return rffRes.noise + yMinObj + (yMaxObj - yMinObj)/2; 
            }, 0.6);
        }
        
        if (this.config.showTotal) {
            drawCurve('#00ff88', (u) => {
                const rffRes = this.rff.noiseAndGrad(u * range, 0, this.t, amp);
                return fnInfo.fn(u * range) + rffRes.noise;
            }, 1.0);
        }

        // Draw Particles
        this.particles.forEach(p => {
            const rffRes = this.rff.noiseAndGrad(p.x * range, 0, this.t, amp);
            const totalY = fnInfo.fn(p.x * range) + rffRes.noise;
            
            ctx.beginPath();
            ctx.arc(toX(p.x), toY(totalY), 6, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.shadowBlur = 10;
            ctx.shadowColor = '#00ff88';
            ctx.fill();
            ctx.shadowBlur = 0;
            
            // Vector (u-space)
            const gObj = fnInfo.grad(p.x * range);
            const gNoise = rffRes.gradX;
            const totalGradNorm = (gObj + gNoise) * range;
            
            ctx.beginPath();
            ctx.moveTo(toX(p.x), toY(totalY));
            ctx.lineTo(toX(p.x - totalGradNorm * 0.05), toY(totalY)); 
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.stroke();
        });
    }

    animate() {
        if (this.isPlaying) {
            this.update();
        }
        this.render();
        requestAnimationFrame(this.animate);
    }
}

// Start app
window.explorer = new Explorer1D();
