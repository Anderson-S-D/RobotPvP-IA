package com.udistrital.robotpvp.ai

import android.content.Context
import org.json.JSONObject
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.tanh
import kotlin.random.Random

/**
 * Cerebro del robot rojo: red de cabezas compartidas.
 *
 *   Entrada 8 -> Oculta 16 (tanh) -> Actor 4 (softmax) + Crítico 1 (lineal)
 *   Parámetros: (8·16+16) + (16·4+4) + (16·1+1) = 229
 *
 * Los pesos se entrenaron en Python y se leen de assets/robot_ac_model.json.
 * En el teléfono SOLO hay inferencia: la red no aprende durante la partida.
 */

// Componente de Descubrimiento (memoria episódica t−1): solo recuerda la última decisión.
class DiscoveryMemory(
    private val codificaciones: FloatArray,
    private val codificacionNinguna: Float,
) {
    var ultimaAccion: Int = -1     // -1 = ninguna
    var ultimoDelta: Float = 0f

    // Entradas 6 y 7 de la red: [codificación de a_{t-1}, tanh(δ_{t-1})].
    fun entradas(): FloatArray {
        val codigo = if (ultimaAccion == -1) codificacionNinguna else codificaciones[ultimaAccion]
        return floatArrayOf(codigo, tanh(ultimoDelta))
    }

    // Tras cada decisión se guarda (a_t, δ_t) para el siguiente estado.
    fun anotar(accion: Int, delta: Float) { ultimaAccion = accion; ultimoDelta = delta }

    // Al empezar cada episodio la memoria vuelve a (ninguna, 0).
    fun reiniciar() = anotar(-1, 0f)
}

class RobotBrain(textoJson: String) {

    // En la app se usa este constructor; el de texto permite probar sin Android.
    constructor(context: Context) : this(
        context.assets.open("robot_ac_model.json").bufferedReader().use { it.readText() }
    )

    // Pesos de la red (y = W·x + b, con W de forma [salidas][entradas]).
    private val json = JSONObject(textoJson)
    private val pesos = json.getJSONObject("weights")
    private val W1 = leerMatriz(pesos, "W1")
    private val b1 = leerVector(pesos, "b1")
    private val Wa = leerMatriz(pesos, "Wa")
    private val ba = leerVector(pesos, "ba")
    private val Wc = leerMatriz(pesos, "Wc")
    private val bc = leerVector(pesos, "bc")

    // Constantes del bloque "meta" del JSON (no se duplican a mano).
    private val meta = json.getJSONObject("meta")
    private val recompensa = meta.getJSONObject("reward")
    val accionesCadaNFrames = meta.getInt("action_repeat")
    val velocidad = meta.getDouble("agent_speed").toFloat()
    val gamma = meta.getDouble("gamma").toFloat()
    val coefAcercamiento = recompensa.getDouble("approach_coef").toFloat()
    val penalizacionPaso = recompensa.getDouble("step_penalty").toFloat()  // ya es negativa
    private val velocidadMinimaJugador = meta.getDouble("theta_zero_speed").toFloat()

    val memoria = DiscoveryMemory(
        leerVector(meta, "action_encoding"),
        meta.getDouble("no_action_encoding").toFloat(),
    )

    // Capa de entrada (8): las 6 de geometría normalizadas como en AGENTS.md + las 2 de memoria.
    // La velocidad del robot rojo es la de su último movimiento (atan2(0, 0) = 0 si aún no se mueve).
    fun observar(
        xRobot: Float, yRobot: Float, vxRobot: Float, vyRobot: Float,
        xJugador: Float, yJugador: Float, vxJugador: Float, vyJugador: Float,
        ancho: Float, alto: Float,
    ): FloatArray {
        val quieto = hypot(vxJugador, vyJugador) < velocidadMinimaJugador
        val recuerdo = memoria.entradas()
        return floatArrayOf(
            (2f * xRobot / ancho - 1f).coerceIn(-1f, 1f),
            (2f * yRobot / alto - 1f).coerceIn(-1f, 1f),
            atan2(vyRobot, vxRobot) / PI.toFloat(),
            (2f * xJugador / ancho - 1f).coerceIn(-1f, 1f),
            (2f * yJugador / alto - 1f).coerceIn(-1f, 1f),
            if (quieto) 0f else atan2(vyJugador, vxJugador) / PI.toFloat(),
            recuerdo[0],
            recuerdo[1],
        )
    }

    // Capa oculta h = tanh(W1·x + b1); Actor = softmax(Wa·h + ba); Crítico V = Wc·h + bc.
    // Devuelve (probabilidades, V). Los bucles for muestran la multiplicación de matrices.
    fun pensar(x: FloatArray): Pair<FloatArray, Float> {
        // Capa oculta (16 neuronas)
        val h = FloatArray(W1.size)
        for (i in W1.indices) {
            var suma = b1[i]
            for (j in x.indices) suma += W1[i][j] * x[j]
            h[i] = tanh(suma)
        }

        // Actor: 4 salidas en orden [UP, DOWN, LEFT, RIGHT]
        val probabilidades = FloatArray(Wa.size)
        var total = 0f
        for (i in Wa.indices) {
            var suma = ba[i]
            for (j in h.indices) suma += Wa[i][j] * h[j]
            probabilidades[i] = exp(suma)
            total += probabilidades[i]
        }
        for (i in probabilidades.indices) probabilidades[i] /= total

        // Crítico: 1 salida lineal
        var valor = bc[0]
        for (j in h.indices) valor += Wc[0][j] * h[j]

        return Pair(probabilidades, valor)
    }

    // "El piloto lanza un dado con los pesos de sus probabilidades" (muestreo).
    // Para usar argmax: return probabilidades.indices.maxBy { probabilidades[it] }
    fun elegirAccion(probabilidades: FloatArray): Int {
        var dado = Random.nextFloat()
        for (i in probabilidades.indices) {
            dado -= probabilidades[i]
            if (dado <= 0f) return i
        }
        return probabilidades.lastIndex  // por si el redondeo deja un resto mínimo
    }

    // Ayuda: lee una matriz [salidas][entradas] del JSON.
    private fun leerMatriz(padre: JSONObject, nombre: String): Array<FloatArray> {
        val filas = padre.getJSONArray(nombre)
        return Array(filas.length()) { i ->
            val fila = filas.getJSONArray(i)
            FloatArray(fila.length()) { j -> fila.getDouble(j).toFloat() }
        }
    }

    // Ayuda: lee un vector del JSON.
    private fun leerVector(padre: JSONObject, nombre: String): FloatArray {
        val valores = padre.getJSONArray(nombre)
        return FloatArray(valores.length()) { i -> valores.getDouble(i).toFloat() }
    }
}

/*
 * Cómo encaja con el informe
 *  - Figura 1 (bucle de memoria): anotar() guarda (a_t, δ_t) y observar() los inyecta en las entradas 6–7.
 *  - Figura 2 (capas): pensar() recorre 8 -> 16 (tanh) -> Actor 4 (softmax) + Crítico 1 (lineal).
 *  - Piloto = Actor: da las probabilidades de cada dirección y elegirAccion() decide con ellas.
 *  - Copiloto = Crítico: V(s) no mueve el robot; evalúa la situación y con ella se calcula δ.
 */
