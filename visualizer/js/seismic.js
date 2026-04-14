class SeismicEngine {
  constructor(fnConfig, rffField, options = {}) {
    this.fnConfig = fnConfig;
    this.fn = fnConfig.fn;
    this.grad = fnConfig.grad;
    this.rff = rffField;
    this.range = fnConfig.range;
    
    this.nParticles = options.nParticles || 10;
    this.dt = options.dt || 0.01;
    this.noiseAmplitude = options.noiseAmplitude || 15.0;
    this.nCycles = options.nCycles || 10;
    this.nSteps = options.nSteps || 2000;
    
    this.reset();
  }
  
  reset(seedOffset = 0) {
    this.step = 0;
    this.t = 0;
    this.dtNoise = (this.nCycles * Math.PI) / this.nSteps;
    
    const rng = mulberry32(123 + seedOffset);
    
    this.particles = [];
    for (let i = 0; i < this.nParticles; i++) {
      this.particles.push({
        x: (rng() * 2 - 1) * this.range,
        y: (rng() * 2 - 1) * this.range,
      });
    }
    
    this.bestVal = Infinity;
    this.bestPos = { x: 0, y: 0 };
    this.bestHistory = [];
    this.trails = this.particles.map(p => [{x: p.x, y: p.y}]);
    
    this._updateBest();
  }
  
  tick() {
    if (this.step >= this.nSteps) return false;
    
    const freq = 2.0;
    const amp = this.noiseAmplitude * Math.sin(this.t * freq);
    
    for (let i = 0; i < this.nParticles; i++) {
      const p = this.particles[i];
      
      const [fgx, fgy] = this.grad(p.x, p.y);
      const rffResult = this.rff.noiseAndGrad(p.x, p.y, this.t, amp);
      
      p.x -= this.dt * (fgx + rffResult.gradX);
      p.y -= this.dt * (fgy + rffResult.gradY);
      
      p.x = Math.max(-this.range, Math.min(this.range, p.x));
      p.y = Math.max(-this.range, Math.min(this.range, p.y));
      
      this.trails[i].push({ x: p.x, y: p.y });
      if (this.trails[i].length > 100) this.trails[i].shift();
    }
    
    this.t += this.dtNoise;
    this.step++;
    this._updateBest();
    return true;
  }
  
  _updateBest() {
    for (const p of this.particles) {
      const val = this.fn(p.x, p.y);
      if (val < this.bestVal) {
        this.bestVal = val;
        this.bestPos = { x: p.x, y: p.y };
      }
    }
    this.bestHistory.push(this.bestVal);
  }
  
  get currentAmplitude() {
    return this.noiseAmplitude * Math.sin(this.t * 2.0);
  }
  
  get currentCycle() {
    return Math.floor(this.t / Math.PI);
  }
  
  get progress() {
    return this.step / this.nSteps;
  }
}