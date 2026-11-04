# Tabla de evaluación — RobotPVP

- Arena fija: 1080 × 1920 px.
- Episodios por duración y política: 30.
- Semillas de episodio: 12345 a 12374 (idénticas para todos los agentes).
- Modo de acción AC: muestreo.

## Resumen general

| Agente | Episodios | Capturas | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Desv. t. captura (s) | Distancia media (px) | Decisiones por episodio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Objetivo aleatorio | 180 | 155 | 86.1 % | 12.537 | 8.750 | 10.283 | 740.9 | 247.3 |
| Acción aleatoria | 180 | 86 | 47.8 % | 19.055 | 16.650 | 14.066 | 873.7 | 474.2 |
| AC sin descubrimiento | 180 | 180 | 100.0 % | 1.670 | 1.708 | 0.865 | 409.5 | 25.4 |
| AC con descubrimiento | 180 | 180 | 100.0 % | 1.698 | 1.817 | 0.893 | 410.6 | 25.8 |
| Persecución voraz | 180 | 180 | 100.0 % | 1.245 | 1.167 | 0.696 | 432.7 | 19.1 |

## Detalle por duración y política

| Agente | Duración (s) | Política del evasor | Episodios | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Distancia media (px) | Decisiones por episodio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Objetivo aleatorio | 30 | flee | 30 | 66.7 % | 12.847 | 12.250 | 834.4 | 278.7 |
| Objetivo aleatorio | 30 | tilt_random | 30 | 90.0 % | 8.554 | 6.367 | 649.8 | 160.9 |
| Objetivo aleatorio | 45 | flee | 30 | 80.0 % | 16.413 | 17.717 | 830.4 | 332.3 |
| Objetivo aleatorio | 45 | tilt_random | 30 | 100.0 % | 11.126 | 7.392 | 650.4 | 167.3 |
| Objetivo aleatorio | 60 | flee | 30 | 80.0 % | 16.413 | 17.717 | 830.1 | 377.3 |
| Objetivo aleatorio | 60 | tilt_random | 30 | 100.0 % | 11.126 | 7.392 | 650.4 | 167.3 |
| Acción aleatoria | 30 | flee | 30 | 26.7 % | 14.006 | 14.592 | 1064.9 | 386.1 |
| Acción aleatoria | 30 | tilt_random | 30 | 50.0 % | 12.962 | 10.133 | 709.5 | 322.4 |
| Acción aleatoria | 45 | flee | 30 | 33.3 % | 18.943 | 18.825 | 1039.5 | 544.8 |
| Acción aleatoria | 45 | tilt_random | 30 | 60.0 % | 16.558 | 11.242 | 711.6 | 419.3 |
| Acción aleatoria | 60 | flee | 30 | 46.7 % | 27.979 | 23.625 | 1012.7 | 676.0 |
| Acción aleatoria | 60 | tilt_random | 30 | 70.0 % | 21.575 | 16.650 | 703.7 | 496.8 |
| AC sin descubrimiento | 30 | flee | 30 | 100.0 % | 1.909 | 2.025 | 422.3 | 29.1 |
| AC sin descubrimiento | 30 | tilt_random | 30 | 100.0 % | 1.431 | 1.217 | 396.7 | 21.8 |
| AC sin descubrimiento | 45 | flee | 30 | 100.0 % | 1.909 | 2.025 | 422.3 | 29.1 |
| AC sin descubrimiento | 45 | tilt_random | 30 | 100.0 % | 1.431 | 1.217 | 396.7 | 21.8 |
| AC sin descubrimiento | 60 | flee | 30 | 100.0 % | 1.909 | 2.025 | 422.3 | 29.1 |
| AC sin descubrimiento | 60 | tilt_random | 30 | 100.0 % | 1.431 | 1.217 | 396.7 | 21.8 |
| AC con descubrimiento | 30 | flee | 30 | 100.0 % | 1.833 | 2.017 | 435.1 | 27.9 |
| AC con descubrimiento | 30 | tilt_random | 30 | 100.0 % | 1.563 | 1.150 | 386.0 | 23.8 |
| AC con descubrimiento | 45 | flee | 30 | 100.0 % | 1.833 | 2.017 | 435.1 | 27.9 |
| AC con descubrimiento | 45 | tilt_random | 30 | 100.0 % | 1.563 | 1.150 | 386.0 | 23.8 |
| AC con descubrimiento | 60 | flee | 30 | 100.0 % | 1.833 | 2.017 | 435.1 | 27.9 |
| AC con descubrimiento | 60 | tilt_random | 30 | 100.0 % | 1.563 | 1.150 | 386.0 | 23.8 |
| Persecución voraz | 30 | flee | 30 | 100.0 % | 1.352 | 1.267 | 442.4 | 20.6 |
| Persecución voraz | 30 | tilt_random | 30 | 100.0 % | 1.139 | 0.933 | 423.0 | 17.5 |
| Persecución voraz | 45 | flee | 30 | 100.0 % | 1.352 | 1.267 | 442.4 | 20.6 |
| Persecución voraz | 45 | tilt_random | 30 | 100.0 % | 1.139 | 0.933 | 423.0 | 17.5 |
| Persecución voraz | 60 | flee | 30 | 100.0 % | 1.352 | 1.267 | 442.4 | 20.6 |
| Persecución voraz | 60 | tilt_random | 30 | 100.0 % | 1.139 | 0.933 | 423.0 | 17.5 |
