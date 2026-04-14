const FUNCTIONS = {
  rastrigin: {
    name: 'Rastrigin',
    fn: (x, y) => 20 + (x*x - 10*Math.cos(2*Math.PI*x)) + (y*y - 10*Math.cos(2*Math.PI*y)),
    grad: (x, y) => [
      2*x + 20*Math.PI*Math.sin(2*Math.PI*x),
      2*y + 20*Math.PI*Math.sin(2*Math.PI*y)
    ],
    range: 5.12,
    globalMin: [0, 0],
    globalMinVal: 0,
    recommendedAmplitude: 15.0
  },
  schwefel: {
    name: 'Schwefel',
    fn: (x, y) => 418.9829*2 - x*Math.sin(Math.sqrt(Math.abs(x))) - y*Math.sin(Math.sqrt(Math.abs(y))),
    grad: (x, y) => {
      const ax = Math.abs(x);
      const ay = Math.abs(y);
      const sx = Math.sqrt(ax + 1e-30);
      const sy = Math.sqrt(ay + 1e-30);
      return [
        -Math.sin(sx) - (x / (2*sx + 1e-30)) * Math.cos(sx),
        -Math.sin(sy) - (y / (2*sy + 1e-30)) * Math.cos(sy)
      ];
    },
    range: 500,
    globalMin: [420.9687, 420.9687],
    globalMinVal: 0,
    recommendedAmplitude: 1500.0
  }
};