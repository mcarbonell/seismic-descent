class SAEngine {
  constructor(fnConfig, options = {}) {
    this.fn = fnConfig.fn;
    this.range = fnConfig.range;
    this.nSteps = options.nSteps || 2000;
    this.T0 = options.T0 || 10.0;
    this.cooling = options.cooling || 0.999;
    this.stepSize = options.stepSize || 0.3;
    this.reset();
  }
  
  reset(seedOffset = 0) {
    this.step = 0;
    this.T = this.T0;
    
    const rng = mulberry32(456 + seedOffset);
    
    this.pos = {
      x: (rng() * 2 - 1) * this.range,
      y: (rng() * 2 - 1) * this.range,
    };
    this.currentVal = this.fn(this.pos.x, this.pos.y);
    this.bestVal = this.currentVal;
    this.bestPos = { ...this.pos };
    this.bestHistory = [this.bestVal];
    this.trail = [{ ...this.pos }];
    this.rng = rng;
  }
  
  tick() {
    if (this.step >= this.nSteps) return false;
    
    const nx = Math.max(-this.range, Math.min(this.range,
      this.pos.x + gaussianRandom(this.rng) * this.stepSize));
    const ny = Math.max(-this.range, Math.min(this.range,
      this.pos.y + gaussianRandom(this.rng) * this.stepSize));
    const newVal = this.fn(nx, ny);
    const delta = newVal - this.currentVal;
    
    if (delta < 0 || this.rng() < Math.exp(-delta / Math.max(this.T, 1e-10))) {
      this.pos = { x: nx, y: ny };
      this.currentVal = newVal;
    }
    if (this.currentVal < this.bestVal) {
      this.bestVal = this.currentVal;
      this.bestPos = { ...this.pos };
    }
    
    this.T *= this.cooling;
    this.step++;
    this.trail.push({ ...this.pos });
    if (this.trail.length > 200) this.trail.shift();
    this.bestHistory.push(this.bestVal);
    return true;
  }
}