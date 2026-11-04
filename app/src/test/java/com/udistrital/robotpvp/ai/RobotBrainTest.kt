package com.udistrital.robotpvp.ai

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

// Equivalencia con Python: pensar() debe dar las mismas salidas que PyTorch (generadas por export.py).
class RobotBrainTest {

    @Test
    fun pensarCoincideConPyTorch() {
        val cerebro = RobotBrain(File("src/main/assets/robot_ac_model.json").readText())
        val vectores = JSONObject(javaClass.getResource("/robot_ac_test_vectors.json")!!.readText())
        val entradas = vectores.getJSONArray("inputs")
        val probabilidades = vectores.getJSONArray("probs")
        val valores = vectores.getJSONArray("value")
        val tolerancia = 1e-4f

        for (n in 0 until entradas.length()) {
            val fila = entradas.getJSONArray(n)
            val x = FloatArray(fila.length()) { fila.getDouble(it).toFloat() }

            val (pi, v) = cerebro.pensar(x)

            val esperadas = probabilidades.getJSONArray(n)
            for (i in pi.indices) {
                assertEquals("vector $n, prob $i", esperadas.getDouble(i).toFloat(), pi[i], tolerancia)
            }
            assertEquals("vector $n, V", valores.getDouble(n).toFloat(), v, tolerancia)
        }
    }
}
