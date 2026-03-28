import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const { data, error } = await supabase
      .from("quant_jarvis")
      .select("*")
      .order("date", { ascending: false })
      .limit(1);

    if (error) {
      console.error("[API /quant] Supabase error:", error.message);
      return NextResponse.json(
        { error: "데이터 조회 중 오류가 발생했습니다" },
        { status: 500 },
      );
    }

    const row = data?.[0];
    if (!row) {
      return NextResponse.json(null);
    }

    const payload = typeof row.data === "object" && row.data !== null
      ? { ...row.data, date: row.date }
      : { date: row.date };
    return NextResponse.json(payload);
  } catch (err) {
    console.error("[API /quant] error:", err);
    return NextResponse.json(
      { error: "데이터 조회 중 오류가 발생했습니다" },
      { status: 500 },
    );
  }
}
