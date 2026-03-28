import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const ticker = (searchParams.get("ticker") || "").trim();

    if (!ticker) {
      return NextResponse.json({ error: "ticker \ud544\uc218" }, { status: 400 });
    }

    let name = ticker;
    let pick = null;

    // 1) quant_jarvis에서 pick 정보
    const { data: jarvisRows } = await supabase
      .from("quant_jarvis")
      .select("data")
      .order("date", { ascending: false })
      .limit(1);

    const jarvis = jarvisRows?.[0]?.data;
    if (jarvis && typeof jarvis === "object") {
      const picks = (jarvis as Record<string, unknown>).picks as Record<string, unknown> | undefined;
      const pickList = (picks?.picks ?? []) as Record<string, unknown>[];

      const found = pickList.find(
        (p) => String(p.ticker || "") === ticker,
      );
      if (found) {
        name = String(found.name || ticker);
        const signals = Array.isArray(found.signals)
          ? (found.signals as string[])
          : typeof found.signals === "string"
            ? (found.signals as string).split(", ").filter(Boolean)
            : [];
        const reasons = Array.isArray(found.reasons)
          ? (found.reasons as string[])
          : typeof found.reason === "string"
            ? [found.reason as string]
            : [];

        pick = {
          grade: String(found.grade || ""),
          score: typeof found.total_score === "number" ? found.total_score : 0,
          signals,
          reasons,
          entry_price: typeof found.entry_price === "number" ? found.entry_price : 0,
          stop_loss: typeof found.stop_loss === "number" ? found.stop_loss : 0,
          target_price: typeof found.target_price === "number" ? found.target_price : 0,
          n_sources: typeof found.n_sources === "number" ? found.n_sources : signals.length,
        };
      }
    }

    // 2) signals 테이블에서 최근 시그널
    const { data: signals } = await supabase
      .from("signals")
      .select("signal_type, grade, score, entry_price, target_price, stop_price, status, signal_date, return_pct, ticker_name")
      .eq("ticker", ticker)
      .order("signal_date", { ascending: false })
      .limit(10);

    const recentSignals = (signals || []).map((s) => {
      if (!name || name === ticker) name = s.ticker_name || ticker;
      return {
        signal_type: s.signal_type,
        grade: s.grade || "",
        score: s.score || 0,
        entry_price: s.entry_price || 0,
        target_price: s.target_price,
        stop_price: s.stop_price,
        status: s.status || "CLOSED",
        signal_date: s.signal_date || "",
        return_pct: s.return_pct || 0,
      };
    });

    // 3) morning_briefings에서 언급 이력
    const { data: briefings } = await supabase
      .from("morning_briefings")
      .select("date, news_picks")
      .order("date", { ascending: false })
      .limit(30);

    const briefingMentions: { date: string; grade: string; score: number; reason?: string }[] = [];
    if (briefings) {
      for (const row of briefings) {
        if (!Array.isArray(row.news_picks)) continue;
        for (const p of row.news_picks) {
          if (!p || typeof p !== "object") continue;
          const pTicker = String(p.ticker || "");
          const pName = String(p.name || p.title || "");
          if (pTicker === ticker || pName === name) {
            if (!name || name === ticker) name = pName || ticker;
            briefingMentions.push({
              date: row.date,
              grade: String(p.grade || ""),
              score: typeof p.score === "number" ? p.score : 0,
              reason: typeof p.reason === "string" ? p.reason : undefined,
            });
          }
        }
      }
    }

    return NextResponse.json({
      ticker,
      name,
      pick,
      recentSignals,
      briefingMentions,
    });
  } catch (e) {
    console.error("[API /stock] error:", e);
    return NextResponse.json(
      { error: "\uc885\ubaa9 \uc815\ubcf4 \uc870\ud68c \uc911 \uc624\ub958\uac00 \ubc1c\uc0dd\ud588\uc2b5\ub2c8\ub2e4" },
      { status: 500 },
    );
  }
}
