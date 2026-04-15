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
            cyclicDt: false
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
            el.addEventListener(event, (e) => {
                this.config[key] = (el.type === 'checkbox') ? el.checked : parseFloat(e.target.value) || e.target.value;
                if (cb) cb(e.target.value);
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
            val: 0
        }));
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

        // We pre-calculate vertical range for amplitude normalization if not done
        // Standard Rastrigin range is ~40, Schwefel ~800, etc.
        // For simplicity, let's assume amplitude 1.0 = 10% of typical vertical range
        const verticalScale = (this.config.funcKey === 'schwefel') ? 800 :
                             (this.config.funcKey === 'rastrigin') ? 40 :
                             (this.config.funcKey === 'ackley') ? 20 : 10;
        
        for (let s = 0; s < updatesToRun; s++) {
            const noiseAmp = this.config.amplitude * 0.01 * verticalScale;
            const phaseAmp = noiseAmp * Math.sin(this.t * 2.0);
            
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
            });

            this.t += dtNoise;
            this.step++;
        }
        
        const normEpsilon = 0.01; // 1% of normalized domain
        const normMin = fnInfo.globalMin / range;
        const successfulOnes = this.particles.filter(p => Math.abs(p.x - normMin) < normEpsilon);
        const successRate = (successfulOnes.length / this.particles.length * 100).toFixed(0);

        document.getElementById('lblStep').innerText = this.step;
        document.getElementById('lblActiveAmp').innerText = (this.config.amplitude * Math.sin(this.t * 2.0)).toFixed(1) + "%";
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

        // Map x [-range, range] to [0, w]
        // Map y [fMin, fMax] to [h, 0]
        
        // Sampling for Y scale
        let yMin = Infinity, yMax = -Infinity;
        const samples = 100;
        const vals = [];
        for (let i = 0; i <= samples; i++) {
            const u = -1 + (i / samples) * 2;
            const x = u * range;
            const yObj = fnInfo.fn(x);
            vals.push(yObj);
            yMin = Math.min(yMin, yObj);
            yMax = Math.max(yMax, yObj);
        }
        
        // Add some margin to Y
        const yRange = yMax - yMin;
        const pad = yRange * 0.5; 
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

        const verticalScale = (this.config.funcKey === 'schwefel') ? 800 :
                             (this.config.funcKey === 'rastrigin') ? 40 :
                             (this.config.funcKey === 'ackley') ? 20 : 10;
        const noiseAmp = this.config.amplitude * 0.01 * verticalScale;
        const amp = noiseAmp * Math.sin(this.t * 2.0);

        if (this.config.showObj) {
            drawCurve('#555', (u) => fnInfo.fn(u * range), 0.8);
        }
        
        if (this.config.showNoise) {
            drawCurve('#ff6b6b', (u) => {
                const rffRes = this.rff.noiseAndGrad(u * range, 0, this.t, amp);
                return rffRes.noise + yMin + yRange/2; 
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
