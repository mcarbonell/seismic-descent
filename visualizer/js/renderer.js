function viridisColor(t) {
    t = Math.max(0, Math.min(1, t));
    const c0 = [0.267004, 0.004874, 0.329415];
    const c1 = [0.127568, 0.566949, 0.550556];
    const c2 = [0.993248, 0.906157, 0.143936];
    
    let r, g, b;
    if (t < 0.5) {
        const u = t * 2.0;
        r = c0[0] + u * (c1[0] - c0[0]);
        g = c0[1] + u * (c1[1] - c0[1]);
        b = c0[2] + u * (c1[2] - c0[2]);
    } else {
        const u = (t - 0.5) * 2.0;
        r = c1[0] + u * (c2[0] - c1[0]);
        g = c1[1] + u * (c2[1] - c1[1]);
        b = c1[2] + u * (c2[2] - c1[2]);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

class Renderer {
  constructor(canvas, fnConfig) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.width = canvas.width;
    this.height = canvas.height;
    this.fnConfig = fnConfig;
    this.baseImage = null;
    this.offscreenCanvas = document.createElement('canvas');
    this.baseResolution = 100;
    this.offscreenCanvas.width = this.baseResolution;
    this.offscreenCanvas.height = this.baseResolution;
    this.offscreenCtx = this.offscreenCanvas.getContext('2d');
  }
  
  precomputeBase() {
    const res = this.baseResolution;
    const imageData = this.offscreenCtx.createImageData(res, res);
    const range = this.fnConfig.range;
    
    let fMin = Infinity, fMax = -Infinity;
    const values = new Float64Array(res * res);
    
    for (let py = 0; py < res; py++) {
      for (let px = 0; px < res; px++) {
        const x = (px / res) * 2 * range - range;
        const y = (py / res) * 2 * range - range;
        const val = this.fnConfig.fn(x, y);
        values[py * res + px] = val;
        fMin = Math.min(fMin, val);
        fMax = Math.max(fMax, val);
      }
    }
    
    for (let i = 0; i < values.length; i++) {
      const t = (values[i] - fMin) / (fMax - fMin);
      const [r, g, b] = viridisColor(t);
      imageData.data[i * 4]     = r;
      imageData.data[i * 4 + 1] = g;
      imageData.data[i * 4 + 2] = b;
      imageData.data[i * 4 + 3] = 255;
    }
    this.offscreenCtx.putImageData(imageData, 0, 0);
  }
  
  mapCoord(x, y) {
    const range = this.fnConfig.range;
    const px = ((x + range) / (2 * range)) * this.width;
    const py = ((y + range) / (2 * range)) * this.height;
    return [px, py];
  }
  
  renderNoiseOverlay(seismicEngine) {
    const res = 50; 
    const imageData = this.ctx.createImageData(res, res);
    const range = this.fnConfig.range;
    
    let amp = seismicEngine.currentAmplitude;
    let maxNoise = Math.abs(amp) * 2.0; 
    if (maxNoise < 1e-5) maxNoise = 1.0;
    
    let u = (seismicEngine.step % seismicEngine.morphSteps) / seismicEngine.morphSteps;
    const weightA = Math.cos(u * Math.PI / 2);
    const weightB = Math.sin(u * Math.PI / 2);
    
    for (let py = 0; py < res; py++) {
      for (let px = 0; px < res; px++) {
        const x = (px / res) * 2 * range - range;
        const y = (py / res) * 2 * range - range;
        const nA = seismicEngine.fieldA.noiseAndGrad(x, y, 0, amp).noise;
        const nB = seismicEngine.fieldB.noiseAndGrad(x, y, 0, amp).noise;
        const noiseVal = weightA * nA + weightB * nB;
        
        let alpha = Math.min(255, Math.abs(noiseVal / maxNoise) * 150);
        let idx = (py * res + px) * 4;
        
        if (noiseVal > 0) {
            imageData.data[idx] = 255;
            imageData.data[idx+1] = 255;
            imageData.data[idx+2] = 255;
        } else {
            imageData.data[idx] = 0;
            imageData.data[idx+1] = 0;
            imageData.data[idx+2] = 0;
        }
        imageData.data[idx+3] = alpha;
      }
    }
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = res;
    tempCanvas.height = res;
    tempCanvas.getContext('2d').putImageData(imageData, 0, 0);
    
    this.ctx.globalCompositeOperation = 'overlay';
    this.ctx.drawImage(tempCanvas, 0, 0, this.width, this.height);
    this.ctx.globalCompositeOperation = 'source-over';
  }
  
  renderTrails(trails, color = '#00ff88') {
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 1;
    
    for (const trail of trails) {
      if (trail.length < 2) continue;
      this.ctx.beginPath();
      const [startX, startY] = this.mapCoord(trail[0].x, trail[0].y);
      this.ctx.moveTo(startX, startY);
      
      for (let i = 1; i < trail.length; i++) {
        const [px, py] = this.mapCoord(trail[i].x, trail[i].y);
        this.ctx.lineTo(px, py);
      }
      this.ctx.globalAlpha = 0.5;
      this.ctx.stroke();
    }
    this.ctx.globalAlpha = 1.0;
  }
  
  renderParticles(particles, color = '#00ff88', glow = 0) {
    for (const p of particles) {
      const [px, py] = this.mapCoord(p.x, p.y);
      this.ctx.beginPath();
      this.ctx.arc(px, py, 3, 0, 2 * Math.PI);
      this.ctx.fillStyle = color;
      if (glow > 0) {
        this.ctx.shadowBlur = glow;
        this.ctx.shadowColor = color;
      }
      this.ctx.fill();
      this.ctx.shadowBlur = 0;
    }
  }
  
  renderGlobalOptimum() {
    const [gx, gy] = this.fnConfig.globalMin;
    const [px, py] = this.mapCoord(gx, gy);
    this.ctx.beginPath();
    this.ctx.arc(px, py, 5, 0, 2 * Math.PI);
    this.ctx.strokeStyle = '#ffffff';
    this.ctx.lineWidth = 2;
    this.ctx.stroke();
    this.ctx.beginPath();
    this.ctx.moveTo(px - 8, py);
    this.ctx.lineTo(px + 8, py);
    this.ctx.moveTo(px, py - 8);
    this.ctx.lineTo(px, py + 8);
    this.ctx.stroke();
  }
  
  renderFrame(particles, trails, seismicEngine, showNoise, showTrails, showOptimum, color, glow) {
    if (this.baseImage === null) {
      this.precomputeBase();
    }
    this.ctx.drawImage(this.offscreenCanvas, 0, 0, this.width, this.height);
    
    if (showNoise && seismicEngine && Math.abs(seismicEngine.currentAmplitude) > 0.01) {
      this.renderNoiseOverlay(seismicEngine);
    }
    
    if (showTrails) {
      this.renderTrails(trails, color);
    }
    
    this.renderParticles(particles, color, glow);
    
    if (showOptimum) {
      this.renderGlobalOptimum();
    }
  }
}