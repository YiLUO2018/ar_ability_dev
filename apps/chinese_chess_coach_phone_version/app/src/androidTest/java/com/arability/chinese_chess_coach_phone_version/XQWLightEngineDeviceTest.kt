package com.arability.chinese_chess_coach_phone_version

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.arability.chinese_chess_coach_phone_version.chess.Board
import com.arability.chinese_chess_coach_phone_version.chess.XQWLightChessEngine
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 在真机上验证 XQWLight 引擎的正确性与算力（速度）。
 * 结果通过 Log.i 输出到 logcat（tag = XQWLightTest），跑完用 adb logcat -d -s XQWLightTest 查看。
 */
@RunWith(AndroidJUnit4::class)
class XQWLightEngineDeviceTest {

    @Test
    fun engineReturnsLegalMoveOnDevice() {
        val engine = XQWLightChessEngine()

        // 1) 标准开局（红先）：应给出正常开局着法（当头炮/起马等）
        val initial = Board.fromFen(Board.INITIAL_FEN)!!
        val t0 = System.currentTimeMillis()
        val m1 = engine.bestMove(initial)
        val dt1 = System.currentTimeMillis() - t0
        assertNotNull("开局应有走法", m1)
        Log.i(TAG, "开局: ${m1!!.piece.name} (${m1.fromFile},${m1.fromRank})->(${m1.toFile},${m1.toRank}) 耗时=${dt1}ms")
        assertTrue(
            "开局走法应合法",
            initial.generateMoves(initial.sideToMove).any {
                it.fromFile == m1.fromFile && it.fromRank == m1.fromRank &&
                    it.toFile == m1.toFile && it.toRank == m1.toRank
            }
        )

        // 2) 开局（黑先，等价于空着后的局面）：也应有合法走法
        val blackFirst = Board.fromFen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b")!!
        val t1 = System.currentTimeMillis()
        val m2 = engine.bestMove(blackFirst)
        val dt2 = System.currentTimeMillis() - t1
        assertNotNull("黑先应有走法", m2)
        Log.i(TAG, "黑先: ${m2!!.piece.name} (${m2.fromFile},${m2.fromRank})->(${m2.toFile},${m2.toRank}) 耗时=${dt2}ms")
    }

    @Test
    fun engineSpeedBenchmarkOnDevice() {
        // 直接跑 XQWLight 搜索，测真机 CPU 的节点速度（KNPS = 千节点/秒）。
        val pos = xqwlight.Position()
        pos.fromFen(Board.INITIAL_FEN)
        val search = xqwlight.Search(pos, 16)
        val t0 = System.currentTimeMillis()
        val mv = search.searchMain(64, 1500)
        val dt = System.currentTimeMillis() - t0
        Log.i(TAG, "算力: 走法编码=$mv 耗时=${dt}ms KNPS=${search.getKNPS()} (千节点/秒)")
        assertTrue("应在时间预算内返回走法", mv > 0)
    }

    companion object {
        private const val TAG = "XQWLightTest"
    }
}
