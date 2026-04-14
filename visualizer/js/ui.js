class ChartRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.width = canvas.width;
    this.height = canvas.height;
    this.padding = { top: 20, right: 20, bottom: 30, left: 50 };
  }
  
  render(seismicHistory, saHistory, maxSteps) {
    this.ctx.clearRect(0, 0, this.width, this.height);
    
    this.ctx.strokeStyle = '#2a2a4a';
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.moveTo(this.padding.left, this.padding.top);
    this.ctx.lineTo(this.padding.left, this.height - this.padding.bottom);
    this.ctx.lineTo(this.width - this.padding.right, this.height - this.padding.bottom);
    this.ctx.stroke();
    
    let maxVal = -Infinity;
    let minVal = Infinity;
    
    for (let val of seismicHistory) {
      if (val > maxVal) maxVal = val;
      if (val < minVal) minVal = val;
    }
    for (let val of saHistory) {
      if (val > maxVal) maxVal = val;
      if (val < minVal) minVal = val;
    }
    
    if (minVal <= 0) minVal = 1e-5;
    if (maxVal <= minVal) maxVal = minVal + 1;
    
    const logMin = Math.log10(minVal);
    const logMax = Math.log10(maxVal);
    const logRange = logMax - logMin || 1;
    
    const plotWidth = this.width - this.padding.left - this.padding.right;
    const plotHeight = this.height - this.padding.top - this.padding.bottom;
    
    const mapPoint = (step, val) => {
      let logVal = Math.log10(Math.max(val, 1e-10));
      const x = this.padding.left + (step / maxSteps) * plotWidth;
      const y = this.padding.top + plotHeight - ((logVal - logMin) / logRange) * plotHeight;
      return [x, y];
    };
    
    this.ctx.fillStyle = '#8888a0';
    this.ctx.font = '10px JetBrains Mono';
    this.ctx.textAlign = 'right';
    this.ctx.textBaseline = 'middle';
    
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const tickLog = logMin + (i / yTicks) * logRange;
      const tickVal = Math.pow(10, tickLog);
      const y = this.padding.top + plotHeight - (i / yTicks) * plotHeight;
      this.ctx.fillText(tickVal.toExponential(1), this.padding.left - 5, y);
    }
    
    const drawLine = (history, color) => {
      if (history.length === 0) return;
      this.ctx.beginPath();
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = 2;
      const [startX, startY] = mapPoint(0, history[0]);
      this.ctx.moveTo(startX, startY);
      for (let i = 1; i < history.length; i++) {
        const [x, y] = mapPoint(i, history[i]);
        this.ctx.lineTo(x, y);
      }
      this.ctx.stroke();
    };
    
    drawLine(seismicHistory, '#00ff88');
    drawLine(saHistory, '#ff6b6b');
  }
}