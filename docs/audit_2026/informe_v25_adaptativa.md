# v25: amplitud adaptativa por régimen detectado (coherencia vs estancamiento)

> Trials: 31 | Budget: 3000 | dims [5, 10, 20] | ORF | init interna pareada (x0=None, misma seed por trial). amp_lo=0.5·√(1/D), amp_hi=0.5 abs. v25 = champion v23, único cambio: política de amplitud.

## Rank medio por celda (1 = mejor en todas)

| variante | rank |
|---|---|
| v24_ref1 | 1.69 |
| v25c_coherence | 2.22 |
| v25s_stagnation | 3.25 |
| v24_ref5 | 3.58 |
| v23_abs | 4.25 |

## Medianas por celda (★ = Wilcoxon Holm del v25 indicado vs v24_ref1)

| celda | clase | v23_abs | v24_ref5 | v24_ref1 | v25c_coherence | v25s_stagnation |
|---|---|---|---|---|---|---|
| ackley_10d | decept. | 14.85 | 12.45 | 5.246 | 13.43*** | 14.06*** |
| ackley_20d | decept. | 18.62 | 12.86 | 4.796 | 17.06*** | 18.13*** |
| ackley_5d | decept. | 7.762 | 7.762 | 4.243 | 6.634*** | 6.938*** |
| dixon_price_10d | valle | 3.33 | 2.464 | 0.74 | 0.8231*** | 1.703*** |
| dixon_price_20d | valle | 21.03 | 7.936 | 1.074 | 1.967*** | 7.986*** |
| dixon_price_5d | valle | 0.7864 | 0.7864 | 0.2273 | 0.3186ns | 0.6987*** |
| griewank_10d | valle | 1.105 | 1.042 | 0.9085 | 0.9272ns | 1.061*** |
| griewank_20d | valle | 1.55 | 1.145 | 0.5523 | 0.8892*** | 1.082*** |
| griewank_5d | valle | 1.012 | 1.012 | 0.7643 | 0.9058ns | 0.982*** |
| levy_10d | decept. | 1.753 | 1.822 | 4.946 | 1.418*** | 1.613*** |
| levy_20d | decept. | 5.322 | 6.276 | 17.98 | 5.903*** | 5.22*** |
| levy_5d | decept. | 0.1708 | 0.1708 | 0.01636 | 0.1356* | 0.1421** |
| michalewicz_10d | decept. | -3.669 | -3.889 | -4.379 | -3.896* | -3.888*** |
| michalewicz_20d | decept. | -4.783 | -5.087 | -5.832 | -4.84*** | -4.79*** |
| michalewicz_5d | decept. | -2.977 | -2.977 | -3.549 | -2.949*** | -3.012*** |
| rastrigin_10d | decept. | 36.43 | 27.32 | 21.42 | 28.53*** | 31.98*** |
| rastrigin_20d | decept. | 87.38 | 75.99 | 77.06 | 86.4* | 93.1*** |
| rastrigin_5d | decept. | 7.677 | 7.677 | 4.716 | 6.806ns | 7.238*** |
| rosenbrock_10d | valle | 22.88 | 16.69 | 5.146 | 6.959*** | 14.59*** |
| rosenbrock_20d | valle | 77.27 | 38.88 | 16 | 18.38** | 36.81*** |
| rosenbrock_5d | valle | 3.914 | 3.914 | 0.4588 | 0.8232* | 3.351*** |
| schwefel_10d | decept. | 1225 | 1164 | 1013 | 1121* | 1176*** |
| schwefel_20d | decept. | 3519 | 3112 | 2713 | 3370*** | 3192*** |
| schwefel_5d | decept. | 300.7 | 300.7 | 335.7 | 245.3ns | 267.4ns |
| sphere_10d | valle | 0.03249 | 0.02215 | 0.0003861 | 0.001184*** | 0.005059*** |
| sphere_20d | valle | 0.1693 | 0.04313 | 0.0006711 | 0.001479*** | 0.02289*** |
| sphere_5d | valle | 0.002551 | 0.002551 | 0.000232 | 0.0003077ns | 0.001655*** |
| styblinski_tang_10d | decept. | -371.3 | -372 | -349.2 | -373.4*** | -370.7*** |
| styblinski_tang_20d | decept. | -705.5 | -697 | -684.4 | -703.9ns | -696.3ns |
| styblinski_tang_5d | decept. | -195.7 | -195.7 | -181.7 | -195.8** | -195.7** |
| trid_10d | valle | -129.6 | -170.1 | -208.4 | -207.6** | -197.3*** |
| trid_20d | valle | 2132 | -57.75 | -1447 | -1350*** | 149.7*** |
| trid_5d | valle | -29.71 | -29.71 | -29.98 | -29.97ns | -29.9*** |
| zakharov_10d | valle | 5.862 | 3.614 | 0.5282 | 3.124*** | 4.931*** |
| zakharov_20d | valle | 94.11 | 28.56 | 9.182 | 67.16*** | 87.62*** |
| zakharov_5d | valle | 0.1146 | 0.1146 | 0.0308 | 0.04946* | 0.1007*** |
