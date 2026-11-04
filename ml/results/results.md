# Resultados — RobotPVP

## Configuración

- Arena fija: 1080 × 1920 px.
- Duraciones evaluadas: 30, 45, 60 s.
- Políticas de evasor: flee, tilt_random.
- Episodios por duración y política: 300.
- Semillas de episodio: 12345 a 12644 (idénticas para todos los agentes).
- Semillas de entrenamiento de los agentes AC: 1, 2, 3.
- Checkpoint de los agentes AC: `last.pt` (mismo presupuesto de entrenamiento para todas las semillas).
- Modo de acción de los agentes AC: muestreo de la política.

## Tabla general

Los agentes AC agregan sus resultados sobre las semillas de entrenamiento indicadas. El tiempo de captura se calcula solo sobre episodios capturados.

| Agente | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Distancia media (px) | Decisiones por episodio |
| --- | --- | --- | --- | --- | --- |
| Objetivo aleatorio | 86.9 % | 12.442 | 9.283 | 724.5 | 241.9 |
| Acción aleatoria | 47.0 % | 17.415 | 13.225 | 899.9 | 467.7 |
| AC sin descubrimiento | 100.0 % | 1.779 | 1.650 | 417.8 | 27.0 |
| AC con descubrimiento | 100.0 % | 1.735 | 1.608 | 419.7 | 26.4 |
| Persecución voraz | 100.0 % | 1.226 | 1.158 | 438.8 | 18.8 |

## Resultados por política del evasor

| Agente | Política del evasor | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Distancia media (px) |
| --- | --- | --- | --- | --- | --- |
| Objetivo aleatorio | flee | 77.7 % | 14.119 | 11.267 | 800.5 |
| Objetivo aleatorio | tilt_random | 96.2 % | 11.089 | 8.633 | 648.4 |
| Acción aleatoria | flee | 29.4 % | 18.448 | 17.283 | 1073.0 |
| Acción aleatoria | tilt_random | 64.6 % | 16.944 | 12.250 | 726.7 |
| AC sin descubrimiento | flee | 100.0 % | 1.935 | 1.842 | 435.6 |
| AC sin descubrimiento | tilt_random | 100.0 % | 1.622 | 1.483 | 400.1 |
| AC con descubrimiento | flee | 100.0 % | 1.890 | 1.842 | 437.9 |
| AC con descubrimiento | tilt_random | 100.0 % | 1.579 | 1.417 | 401.6 |
| Persecución voraz | flee | 100.0 % | 1.323 | 1.300 | 450.9 |
| Persecución voraz | tilt_random | 100.0 % | 1.130 | 1.050 | 426.6 |

## Resultados por duración

| Agente | Duración (s) | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Decisiones por episodio |
| --- | --- | --- | --- | --- | --- |
| Objetivo aleatorio | 30 | 80.8 % | 10.342 | 8.300 | 211.9 |
| Objetivo aleatorio | 45 | 87.5 % | 12.303 | 9.333 | 246.2 |
| Objetivo aleatorio | 60 | 92.5 % | 14.410 | 10.233 | 267.7 |
| Acción aleatoria | 30 | 38.0 % | 11.719 | 10.142 | 345.9 |
| Acción aleatoria | 45 | 47.7 % | 16.853 | 13.908 | 473.9 |
| Acción aleatoria | 60 | 55.3 % | 21.812 | 18.342 | 583.2 |
| AC sin descubrimiento | 30 | 100.0 % | 1.779 | 1.650 | 27.0 |
| AC sin descubrimiento | 45 | 100.0 % | 1.779 | 1.650 | 27.0 |
| AC sin descubrimiento | 60 | 100.0 % | 1.779 | 1.650 | 27.0 |
| AC con descubrimiento | 30 | 100.0 % | 1.735 | 1.608 | 26.4 |
| AC con descubrimiento | 45 | 100.0 % | 1.735 | 1.608 | 26.4 |
| AC con descubrimiento | 60 | 100.0 % | 1.735 | 1.608 | 26.4 |
| Persecución voraz | 30 | 100.0 % | 1.226 | 1.158 | 18.8 |
| Persecución voraz | 45 | 100.0 % | 1.226 | 1.158 | 18.8 |
| Persecución voraz | 60 | 100.0 % | 1.226 | 1.158 | 18.8 |

## Agentes AC por semilla de entrenamiento

| Agente | Semilla de entrenamiento | Tasa de captura | Tiempo medio de captura (s) | Tiempo mediano de captura (s) | Distancia media (px) |
| --- | --- | --- | --- | --- | --- |
| AC sin descubrimiento | 1 | 100.0 % | 1.721 | 1.650 | 418.7 |
| AC sin descubrimiento | 2 | 100.0 % | 1.812 | 1.650 | 418.7 |
| AC sin descubrimiento | 3 | 100.0 % | 1.802 | 1.667 | 416.1 |
| AC con descubrimiento | 1 | 100.0 % | 1.707 | 1.567 | 421.4 |
| AC con descubrimiento | 2 | 100.0 % | 1.765 | 1.633 | 417.6 |
| AC con descubrimiento | 3 | 100.0 % | 1.732 | 1.617 | 420.3 |

## Componente de Descubrimiento (ac_disc)

- Muestras de decisiones analizadas: **137,040**.
- Muestras con δ_{t-1} < 0: **55,743**; con δ_{t-1} > 0: **81,297**; con δ_{t-1} = 0: **0**.
- P(repetir la acción anterior | δ_{t-1} < 0) = **55.65 %**.
- P(repetir la acción anterior | δ_{t-1} > 0) = **72.86 %**.
- Diferencia (δ<0 − δ>0) = **-17.21 pp**.

Si el agente usa la memoria de descubrimiento, se espera que P(repetir | δ<0) sea menor que P(repetir | δ>0). El número se reporta tal como sale de las evaluaciones, aunque no favorezca a la propuesta.

## Curvas de aprendizaje

Tasa de captura final (último punto de `progress.csv`, media ± desviación entre semillas):

| Método | Semillas | Tasa de captura final | Desviación |
| --- | --- | --- | --- |
| AC sin descubrimiento | 3 | 100.0 % | 0.0 pp |
| AC con descubrimiento | 3 | 100.0 % | 0.0 pp |

## Notas metodológicas

- Los agentes AC se evalúan con `last.pt` (política final tras el mismo presupuesto de 2 000 000 decisiones en todas las semillas). Así se evita el sesgo de `best.pt`, que con la tasa de captura saturada en 1.0 podía quedar anclado a la primera ventana que alcanzaba el máximo. `--checkpoint-name best.pt` permite evaluar el mejor checkpoint si se desea.
- `random_target` reproduce el comportamiento del robot rojo actual: objetivo aleatorio y velocidad de 12 px/frame. Como la interfaz de evaluación decide cada `action_repeat` frames, la dirección se aproxima con la acción cardinal dominante y el objetivo se renueva cuando la distancia es menor que 12·`action_repeat` px.
- `random_action` muestrea una acción uniforme en {UP, DOWN, LEFT, RIGHT} cada `action_repeat` frames.
- `greedy_chase` elige el eje con mayor diferencia de posición y avanza hacia el jugador; es una referencia superior, no un agente entrenado.
- La distancia media se muestrea al inicio y al final de cada decisión (cada `action_repeat` frames), no en cada frame.
- Las semillas de episodio son idénticas para todos los agentes, pero las trayectorias difieren porque cada agente toma decisiones distintas.
