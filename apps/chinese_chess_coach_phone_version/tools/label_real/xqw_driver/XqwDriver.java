package xqwlight;

import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * 批量引擎驱动：stdin 每行一个 FEN（红方行棋），stdout 每行 "ff fr tf tr" 或 "none"。
 * 与 App 的 XQWLightChessEngine 桥接同参数：searchMain(64, 1000)。
 * 根节点无合法走法时 searchRoot 会触发 historyIndex(0) 负下标越界（已知坑），
 * 这里用 catch Throwable 兜住输出 none。
 */
public class XqwDriver {
    public static void main(String[] args) throws Exception {
        Position position = new Position();
        Search search = new Search(position, 16);
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        StringBuilder out = new StringBuilder();
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String fen = line.endsWith(" w") || line.endsWith(" b") ? line : line + " w";
            int mv;
            try {
                position.fromFen(fen);
                mv = search.searchMain(64, 1000);
            } catch (Throwable t) {
                mv = 0;
            }
            if (mv <= 0) {
                out.append("none\n");
            } else {
                int src = mv & 0xff;
                int dst = mv >>> 8;
                out.append((src & 0x0f) - Position.FILE_LEFT).append(' ')
                   .append((src >> 4) - Position.RANK_TOP).append(' ')
                   .append((dst & 0x0f) - Position.FILE_LEFT).append(' ')
                   .append((dst >> 4) - Position.RANK_TOP).append('\n');
            }
            System.out.print(out);
            out.setLength(0);
            System.out.flush();
        }
    }
}
