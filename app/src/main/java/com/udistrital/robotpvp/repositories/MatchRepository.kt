package com.udistrital.robotpvp.repositories

import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.database.DataSnapshot
import com.google.firebase.database.DatabaseError
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ValueEventListener
import com.udistrital.robotpvp.data.enums.GameDifficulty
import com.udistrital.robotpvp.data.models.GameModel
import com.udistrital.robotpvp.data.models.MatchModel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

class MatchRepository {
    private val database = FirebaseDatabase.getInstance()
    private val matchesRef = database.getReference("matches")
    private val auth = FirebaseAuth.getInstance()

    companion object {
        const val GAME_CODE_LENGTH = 5
        private val CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    }

    fun generateGameCode(): String {
        return (1..GAME_CODE_LENGTH).map { CODE_CHARS.random() }.joinToString("")
    }

    suspend fun createMatch(difficulty: GameDifficulty = GameDifficulty.EASY): Result<MatchModel> {
        return try {
            val uid = auth.currentUser?.uid?.takeIf { it.isNotBlank() } ?: "user1"
            var code = generateGameCode()
            // Ensure unique code
            while (matchesRef.child(code).get().await().exists()) {
                code = generateGameCode()
            }

            val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.getDefault())
            val now = dateFormat.format(Date())

            val match = MatchModel(
                code = code,
                player1Uid = uid,
                player2Uid = "",
                startAt = "",
                turnPlayerUid = uid,
                winnerPlayerUid = "",
                difficulty = difficulty.name,
                gameDurationSeconds = difficulty.durationSeconds,
                timeRemaining = difficulty.durationSeconds,
                createdAt = now,
                finishedAt = "",
                game = GameModel(),
                status = com.udistrital.robotpvp.data.enums.MatchStatus.WAITING.name,
                isGameOver = false
            )

            matchesRef.child(code).setValue(match).await()
            Result.success(match)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun joinMatch(code: String): Result<MatchModel> {
        return try {
            val uid = auth.currentUser?.uid?.takeIf { it.isNotBlank() } ?: "user1"
            val snapshot = matchesRef.child(code.uppercase()).get().await()
            if (!snapshot.exists()) {
                return Result.failure(Exception("Partida no encontrada"))
            }
            val match = snapshot.getValue(MatchModel::class.java)
            if (match == null) {
                return Result.failure(Exception("Error al leer la partida"))
            }
            if (match.player2Uid.isNotEmpty()) {
                return Result.failure(Exception("La partida ya está llena"))
            }
            if (match.player1Uid == uid) {
                return Result.failure(Exception("No puedes unirte a tu propia partida"))
            }

            matchesRef.child(code.uppercase()).child("player2Uid").setValue(uid).await()
            matchesRef.child(code.uppercase()).child("status").setValue(com.udistrital.robotpvp.data.enums.MatchStatus.PLAYING.name).await()
            matchesRef.child(code.uppercase()).child("startAt")
                .setValue(SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.getDefault()).format(Date()))
                .await()

            val updatedSnapshot = matchesRef.child(code.uppercase()).get().await()
            val updatedMatch = updatedSnapshot.getValue(MatchModel::class.java)
            Result.success(updatedMatch ?: match.copy(player2Uid = uid, status = com.udistrital.robotpvp.data.enums.MatchStatus.PLAYING.name))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    fun observeMatch(code: String): Flow<MatchModel?> = callbackFlow {
        val listener = object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                val match = snapshot.getValue(MatchModel::class.java)
                trySend(match)
            }

            override fun onCancelled(error: DatabaseError) {
                close(error.toException())
            }
        }
        matchesRef.child(code.uppercase()).addValueEventListener(listener)
        awaitClose {
            matchesRef.child(code.uppercase()).removeEventListener(listener)
        }
    }

    suspend fun leaveMatch(code: String) {
        try {
            matchesRef.child(code.uppercase()).removeValue().await()
        } catch (e: Exception) {
        }
    }
}