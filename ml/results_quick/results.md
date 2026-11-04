# Resultados — RobotPVP

## Configuración

- Arena fija: 1080 × 1920 px.
- Duraciones evaluadas: 30, 45, 60 s.
- Políticas de evasor: flee, tilt_random.
- Episodios por duración y política: 30.
- Semillas de episodio: 12345 a 12374 (idénticas para todos los agentes).
- Semillas de entrenamiento de los agentes AC: 1.
- Checkpoint de los agentes AC: `last.pt` (mismo presupuesto de entrenamiento para todas las semillas).
- Modo de acción de los agentes AC: muestreo de la política.

## Tabla general

Los agentes AC agregan sus resultados sobre las semillas de entrenamiento indicadas. El tiempo de captura se calcula solo sobre episodios capturados.

| Agente | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Distancia media (px) | Decisiones por episodio |
| --- | --- | --- | --- | --- | --- |
| Objetivo aleatorio | 86.1 % | 12.537 | 8.750 | 740.9 | 247.3 |
| Acción aleatoria | 47.8 % | 19.055 | 16.650 | 873.7 | 474.2 |
| AC sin descubrimiento | 100.0 % | 1.670 | 1.708 | 409.5 | 25.4 |
| AC con descubrimiento | 100.0 % | 1.698 | 1.817 | 410.6 | 25.8 |
| Persecución voraz | 100.0 % | 1.245 | 1.167 | 432.7 | 19.1 |

## Resultados por política del evasor

| Agente | Política del evasor | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Distancia media (px) |
| --- | --- | --- | --- | --- | --- |
| Objetivo aleatorio | flee | 75.6 % | 15.364 | 17.100 | 831.6 |
| Objetivo aleatorio | tilt_random | 96.7 % | 10.328 | 7.333 | 650.2 |
| Acción aleatoria | flee | 35.6 % | 21.662 | 19.800 | 1039.0 |
| Acción aleatoria | tilt_random | 60.0 % | 17.510 | 11.242 | 708.3 |
| AC sin descubrimiento | flee | 100.0 % | 1.909 | 2.025 | 422.3 |
| AC sin descubrimiento | tilt_random | 100.0 % | 1.431 | 1.217 | 396.7 |
| AC con descubrimiento | flee | 100.0 % | 1.833 | 2.017 | 435.1 |
| AC con descubrimiento | tilt_random | 100.0 % | 1.563 | 1.150 | 386.0 |
| Persecución voraz | flee | 100.0 % | 1.352 | 1.267 | 442.4 |
| Persecución voraz | tilt_random | 100.0 % | 1.139 | 0.933 | 423.0 |

## Resultados por duración

| Agente | Duración (s) | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Decisiones por episodio |
| --- | --- | --- | --- | --- | --- |
| Objetivo aleatorio | 30 | 78.3 % | 10.380 | 7.683 | 219.8 |
| Objetivo aleatorio | 45 | 90.0 % | 13.476 | 9.200 | 249.8 |
| Objetivo aleatorio | 60 | 90.0 % | 13.476 | 9.200 | 272.3 |
| Acción aleatoria | 30 | 38.3 % | 13.325 | 11.017 | 354.2 |
| Acción aleatoria | 45 | 46.7 % | 17.410 | 14.058 | 482.0 |
| Acción aleatoria | 60 | 58.3 % | 24.136 | 19.800 | 586.4 |
| AC sin descubrimiento | 30 | 100.0 % | 1.670 | 1.708 | 25.4 |
| AC sin descubrimiento | 45 | 100.0 % | 1.670 | 1.708 | 25.4 |
| AC sin descubrimiento | 60 | 100.0 % | 1.670 | 1.708 | 25.4 |
| AC con descubrimiento | 30 | 100.0 % | 1.698 | 1.817 | 25.8 |
| AC con descubrimiento | 45 | 100.0 % | 1.698 | 1.817 | 25.8 |
| AC con descubrimiento | 60 | 100.0 % | 1.698 | 1.817 | 25.8 |
| Persecución voraz | 30 | 100.0 % | 1.245 | 1.167 | 19.1 |
| Persecución voraz | 45 | 100.0 % | 1.245 | 1.167 | 19.1 |
| Persecución voraz | 60 | 100.0 % | 1.245 | 1.167 | 19.1 |

## Agentes AC por semilla de entrenamiento

| Agente | Semilla de entrenamiento | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Distancia media (px) |
| --- | --- | --- | --- | --- | --- |
| AC sin descubrimiento | 1 | 100.0 % | 1.670 | 1.708 | 409.5 |
| AC con descubrimiento | 1 | 100.0 % | 1.698 | 1.817 | 410.6 |

## Componente de Descubrimiento (ac_disc)

- Muestras de decisiones analizadas: **4,470**.
- Muestras con δ_{t-1} < 0: **1,701**; con δ_{t-1} > 0: **2,769**; con δ_{t-1} = 0: **0**.
- P(repetir la acción anterior | δ_{t-1} < 0) = **55.03 %**.
- P(repetir la acción anterior | δ_{t-1} > 0) = **72.59 %**.
- Diferencia (δ<0 − δ>0) = **-17.56 pp**.

Si el agente usa la memoria de descubrimiento, se espera que P(repetir | δ<0) sea menor que P(repetir | δ>0). El número se reporta tal como sale de las evaluaciones, aunque no favorezca a la propuesta.

## Curvas de aprendizaje

Tasa de captura final (último punto de `progress.csv`, media ± desviación entre semillas):

| Método | Semillas | Tasa de captura final | Desviación |
| --- | --- | --- | --- |
| AC sin descubrimiento | 1 | 100.0 % | 0.0 pp |
| AC con descubrimiento | 1 | 100.0 % | 0.0 pp |

## Notas metodológicas

- Los agentes AC se evalúan con `last.pt` (política final tras el mismo presupuesto de 2 000 000 decisiones en todas las semillas). Así se evita el sesgo de `best.pt`, que con la tasa de captura saturada en 1.0 podía quedar anclado a la primera ventana que alcanzaba el máximo. `--checkpoint-name best.pt` permite evaluar el mejor checkpoint si se desea.
- `random_target` reproduce el comportamiento del robot rojo actual: objetivo aleatorio y velocidad de 12 px/frame. Como la interfaz de evaluación decide cada `action_repeat` frames, la dirección se aproxima con la acción cardinal dominante y el objetivo se renueva cuando la distancia es menor que 12·`action_repeat` px.
- `random_action` muestrea una acción uniforme en {UP, DOWN, LEFT, RIGHT} cada `action_repeat` frames.
- `greedy_chase` elige el eje con mayor diferencia de posición y avanza hacia el jugador; es una referencia superior, no un agente entrenado.
- La distancia media se muestrea al inicio y al final de cada decisión (cada `action_repeat` frames), no en cada frame.
- Las semillas de episodio son idénticas para todos los agentes, pero las trayectorias difieren porque cada agente toma decisiones distintas.
