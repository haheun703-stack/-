import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

const VALID_BOT_TYPES = ["QUANT", "DAYTRADING", "ALL"];
const VALID_STATUSES = ["OPEN", "CLOSED", "STOPPED"];

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const botType = searchParams.get("bot_type") || "QUANT";
  const status = searchParams.get("status");
  const rawLimit = parseInt(searchParams.get("limit") || "20", 10);
  const limit = isNaN(rawLimit) ? 20 : Math.min(Math.max(rawLimit, 1), 100);

  if (!VALID_BOT_TYPES.includes(botType)) {
    return NextResponse.json({ error: "잘못된 bot_type" }, { status: 400 });
  }
  if (status && !VALID_STATUSES.includes(status)) {
    return NextResponse.json({ error: "잘못된 status" }, { status: 400 });
  }

  let query = supabase
    .from("signals")
    .select("*")
    .eq("bot_type", botType)
    .order("signal_date", { ascending: false })
    .limit(limit);

  if (status) {
    query = query.eq("status", status);
  }

  const { data, error } = await query;

  if (error) {
    console.error("[API /signals] Supabase error:", error.message);
    return NextResponse.json({ error: "시그널 조회 중 오류가 발생했습니다" }, { status: 500 });
  }

  return NextResponse.json(data || []);
}
