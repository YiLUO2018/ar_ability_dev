package com.arability.face_recognition_phone_version.face

import android.content.Context
import android.content.SharedPreferences

/**
 * 人脸特征库：保存已注册人脸（名字 → 归一化特征向量）。
 * 匹配采用余弦相似度（特征已 L2 归一化，点积即余弦）。
 *
 * 特征通过 [SharedPreferences] 持久化，应用重启后已注册人脸仍然保留。
 */
class FaceDatabase(
    context: Context,
    private val matchThreshold: Float = DEFAULT_THRESHOLD
) {

    data class MatchResult(val name: String, val similarity: Float)

    private class Entry(val name: String, val embedding: FloatArray)

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val entries = mutableListOf<Entry>()

    init {
        loadAll()
    }

    fun register(name: String, embedding: FloatArray) {
        entries.removeAll { it.name == name }
        entries.add(Entry(name, embedding))
        prefs.edit().putString(keyOf(name), serialize(embedding)).apply()
    }

    /**
     * 返回相似度最高的命中结果；若最高相似度低于阈值或无注册人脸，返回 null。
     */
    fun match(embedding: FloatArray): MatchResult? {
        var best: MatchResult? = null
        for (entry in entries) {
            val similarity = cosineSimilarity(embedding, entry.embedding)
            if (similarity >= matchThreshold && (best == null || similarity > best.similarity)) {
                best = MatchResult(entry.name, similarity)
            }
        }
        return best
    }

    /**
     * 返回与库中所有人脸的最高余弦相似度（不设阈值），用于诊断阈值是否合理。
     */
    fun topSimilarity(embedding: FloatArray): Float {
        var best = 0f
        for (entry in entries) {
            val similarity = cosineSimilarity(embedding, entry.embedding)
            if (similarity > best) best = similarity
        }
        return best
    }

    fun size(): Int = entries.size

    fun names(): List<String> = entries.map { it.name }

    private fun loadAll() {
        for ((key, value) in prefs.all) {
            if (!key.startsWith(KEY_PREFIX)) continue
            val embedding = deserialize(value as? String) ?: continue
            entries.add(Entry(key.removePrefix(KEY_PREFIX), embedding))
        }
    }

    private fun keyOf(name: String) = KEY_PREFIX + name

    private fun serialize(embedding: FloatArray): String = embedding.joinToString(",")

    private fun deserialize(s: String?): FloatArray? {
        if (s.isNullOrEmpty()) return null
        return try {
            s.split(',').map { it.toFloat() }.toFloatArray()
        } catch (e: NumberFormatException) {
            null
        }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot
    }

    companion object {
        private const val PREFS_NAME = "face_database"
        private const val KEY_PREFIX = "face_"
        private const val DEFAULT_THRESHOLD = 0.4f
    }
}
