class SeismicEngine {
  constructor(fnConfig, options = {}) {
    this.fnConfig = fnConfig;
    this.fn = fnConfig.fn;
    this.grad = fnConfig.grad;
    this.range = fnConfig.range;
    
    this.nParticles = options.nParticles || 10;
    this.dt = options.dt || 0.01;
    this.noiseAmplitude = options.noiseAmplitude || 15.0;
    this.nSteps = options.nSteps || 2000;
    this.morphSteps = options.morphSteps || 100;
    
    this.reset();
  }
  
  reset(seedOffset = 0) {
    this.step = 0;
    
    this.seedA = 1 + seedOffset * 10;
    this.seedB = 2 + seedOffset * 10;
    
    this.fieldA = new RFFField(64, 4, this.seedA, this.range);
    this.fieldB = new RFFField(64, 4, this.seedB, this.range);
    
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
    
    let u = (this.step % this.morphSteps) / this.morphSteps;
    const weightA = Math.cos(u * Math.PI / 2);
    const weightB = Math.sin(u * Math.PI / 2);
    
    const amp = this.currentAmplitude;
    
    for (let i = 0; i < this.nParticles; i++) {
      const p = this.particles[i];
      
      const [fgx, fgy] = this.grad(p.x, p.y);
      const resA = this.fieldA.noiseAndGrad(p.x, p.y, 0, amp);
      const resB = this.fieldB.noiseAndGrad(p.x, p.y, 0, amp);
      
      const noiseGradX = weightA * resA.gradX + weightB * resB.gradX;
      const noiseGradY = weightA * resA.gradY + weightB * resB.gradY;
      
      p.x -= this.dt * (fgx + noiseGradX);
      p.y -= this.dt * (fgy + noiseGradY);
      
      p.x = Math.max(-this.range, Math.min(this.range, p.x));
      p.y = Math.max(-this.range, Math.min(this.range, p.y));
      
      this.trails[i].push({ x: p.x, y: p.y });
      if (this.trails[i].length > 100) this.trails[i].shift();
    }
    
    this.step++;
    
    if (this.step % this.morphSteps === 0) {
        this.seedA = this.seedB;
        this.seedB++;
        this.fieldA = this.fieldB;
        this.fieldB = new RFFField(64, 4, this.seedB, this.range);
    }
    
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
    const decayFactor = 1.0 - (this.step / this.nSteps);
    return this.noiseAmplitude * decayFactor;
  }
  
  get currentCycle() {
    return Math.floor(this.step / this.morphSteps);
  }
  
  get progress() {
    return this.step / this.nSteps;
  }
}