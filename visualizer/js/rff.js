function mulberry32(a) {
    return function() {
      var t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}

function gaussianRandom(rng) {
    let u = 1 - rng();
    let v = rng();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

class RFFField {
  constructor(R = 64, nOctaves = 4, seed = 1, searchRange = 5.12) {
    this.R = R;
    this.nOctaves = nOctaves;
    const rng = mulberry32(seed);
    
    this.omegas = [];
    this.phis = [];
    this.drifts = [];
    
    const scaleFactor = searchRange / 5.12;
    
    for (let o = 0; o < nOctaves; o++) {
      const lengthscale = (scaleFactor * 2.0) / Math.pow(2.0, o);
      const octOmegas = [];
      const octPhis = [];
      const octDrifts = [];
      for (let r = 0; r < R; r++) {
        octOmegas.push({
          x: gaussianRandom(rng) / lengthscale,
          y: gaussianRandom(rng) / lengthscale
        });
        octPhis.push(rng() * 2 * Math.PI);
        octDrifts.push(0.1 + rng() * 0.4);
      }
      this.omegas.push(octOmegas);
      this.phis.push(octPhis);
      this.drifts.push(octDrifts);
    }
  }
  
  noiseAndGrad(x, y, t, amplitude) {
    let noiseVal = 0;
    let gradX = 0, gradY = 0;
    let amp = amplitude;
    const sqrt2R = Math.sqrt(2.0 / this.R);
    
    for (let o = 0; o < this.nOctaves; o++) {
      for (let r = 0; r < this.R; r++) {
        const w = this.omegas[o][r];
        const phi = this.phis[o][r];
        const drift = this.drifts[o][r];
        const angle = w.x * x + w.y * y + t * drift + phi;
        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);
        noiseVal += amp * sqrt2R * cosA;
        gradX -= amp * sqrt2R * sinA * w.x;
        gradY -= amp * sqrt2R * sinA * w.y;
      }
      amp *= 0.5;
    }
    return { noise: noiseVal, gradX, gradY };
  }
}