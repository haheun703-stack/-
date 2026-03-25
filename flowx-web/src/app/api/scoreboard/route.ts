import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const botType = searchParams.get("bot_type") || "QUANT";
  const period = searchParams.get("period") || "30D";

  const { data, error } = await supabase
    .from("scoreboard")
    .select("*")
    .eq("bot_type", botType)
    .eq("period", period)
    .single();

  if (error) {
    // 데이터 없으면 빈 성적표 반환
    if (error.code === "PGRST116") {
      return NextResponse.json({
        bot_type: botType,
        period,
        total_signals: 0,
        win_count: 0,
        lose_count: 0,
        win_rate: 0,
        avg_return_pct: 0,
        avg_win_pct: 0,
        avg_lose_pct: 0,
        best_signal: null,
        worst_signal: null,
      });
    }
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data);
}
