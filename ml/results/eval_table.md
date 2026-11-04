# Tabla de evaluación — RobotPVP

- Arena fija: 1080 × 1920 px.
- Episodios por duración y política: 300.
- Semillas de episodio: 12345 a 12644 (idénticas para todos los agentes).
- Modo de acción AC: muestreo.

## Resumen general

| Agente | Episodios | Capturas | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Desv. t. captura (s) | Distancia media (px) | Decisiones por episodio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Objetivo aleatorio | 1800 | 1565 | 86.9 % | 12.442 | 9.283 | 11.171 | 724.5 | 241.9 |
| Acción aleatoria | 1800 | 846 | 47.0 % | 17.415 | 13.225 | 14.577 | 899.9 | 467.7 |
| AC sin descubrimiento | 5400 | 5400 | 100.0 % | 1.779 | 1.650 | 1.007 | 417.8 | 27.0 |
| AC con descubrimiento | 5400 | 5400 | 100.0 % | 1.735 | 1.608 | 0.985 | 419.7 | 26.4 |
| Persecución voraz | 1800 | 1800 | 100.0 % | 1.226 | 1.158 | 0.665 | 438.8 | 18.8 |

## Detalle por duración y política

| Agente | Duración (s) | Política del evasor | Episodios | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Distancia media (px) | Decisiones por episodio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Objetivo aleatorio | 30 | flee | 300 | 69.7 % | 11.120 | 9.117 | 801.1 | 252.9 |
| Objetivo aleatorio | 30 | tilt_random | 300 | 92.0 % | 9.752 | 8.058 | 649.5 | 170.9 |
| Objetivo aleatorio | 45 | flee | 300 | 78.0 % | 13.708 | 11.308 | 800.8 | 309.1 |
| Objetivo aleatorio | 45 | tilt_random | 300 | 97.0 % | 11.173 | 8.750 | 648.2 | 183.2 |
| Objetivo aleatorio | 60 | flee | 300 | 85.3 % | 16.944 | 14.258 | 799.4 | 349.2 |
| Objetivo aleatorio | 60 | tilt_random | 300 | 99.7 % | 12.240 | 8.967 | 647.7 | 186.3 |
| Acción aleatoria | 30 | flee | 300 | 23.3 % | 12.101 | 10.525 | 1082.4 | 387.4 |
| Acción aleatoria | 30 | tilt_random | 300 | 52.7 % | 11.549 | 10.092 | 735.0 | 304.4 |
| Acción aleatoria | 45 | flee | 300 | 29.3 % | 17.324 | 16.792 | 1072.5 | 553.3 |
| Acción aleatoria | 45 | tilt_random | 300 | 66.0 % | 16.644 | 12.767 | 727.1 | 394.5 |
| Acción aleatoria | 60 | flee | 300 | 35.7 % | 23.526 | 21.333 | 1064.0 | 705.0 |
| Acción aleatoria | 60 | tilt_random | 300 | 75.0 % | 20.996 | 17.267 | 718.1 | 461.5 |
| AC sin descubrimiento | 30 | flee | 900 | 100.0 % | 1.935 | 1.842 | 435.6 | 29.4 |
| AC sin descubrimiento | 30 | tilt_random | 900 | 100.0 % | 1.622 | 1.483 | 400.1 | 24.7 |
| AC sin descubrimiento | 45 | flee | 900 | 100.0 % | 1.935 | 1.842 | 435.6 | 29.4 |
| AC sin descubrimiento | 45 | tilt_random | 900 | 100.0 % | 1.622 | 1.483 | 400.1 | 24.7 |
| AC sin descubrimiento | 60 | flee | 900 | 100.0 % | 1.935 | 1.842 | 435.6 | 29.4 |
| AC sin descubrimiento | 60 | tilt_random | 900 | 100.0 % | 1.622 | 1.483 | 400.1 | 24.7 |
| AC con descubrimiento | 30 | flee | 900 | 100.0 % | 1.890 | 1.842 | 437.9 | 28.7 |
| AC con descubrimiento | 30 | tilt_random | 900 | 100.0 % | 1.579 | 1.417 | 401.6 | 24.0 |
| AC con descubrimiento | 45 | flee | 900 | 100.0 % | 1.890 | 1.842 | 437.9 | 28.7 |
| AC con descubrimiento | 45 | tilt_random | 900 | 100.0 % | 1.579 | 1.417 | 401.6 | 24.0 |
| AC con descubrimiento | 60 | flee | 900 | 100.0 % | 1.890 | 1.842 | 437.9 | 28.7 |
| AC con descubrimiento | 60 | tilt_random | 900 | 100.0 % | 1.579 | 1.417 | 401.6 | 24.0 |
| Persecución voraz | 30 | flee | 300 | 100.0 % | 1.323 | 1.300 | 450.9 | 20.2 |
| Persecución voraz | 30 | tilt_random | 300 | 100.0 % | 1.130 | 1.050 | 426.6 | 17.3 |
| Persecución voraz | 45 | flee | 300 | 100.0 % | 1.323 | 1.300 | 450.9 | 20.2 |
| Persecución voraz | 45 | tilt_random | 300 | 100.0 % | 1.130 | 1.050 | 426.6 | 17.3 |
| Persecución voraz | 60 | flee | 300 | 100.0 % | 1.323 | 1.300 | 450.9 | 20.2 |
| Persecución voraz | 60 | tilt_random | 300 | 100.0 % | 1.130 | 1.050 | 426.6 | 17.3 |
