import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const { data, error } = await supabase
      .from("quant_scenario_dashboard")
      .select("*")
      .order("date", { ascending: false })
      .limit(1);

    if (error) {
      console.error("[API /scenarios] Supabase error:", error.message);
      return NextResponse.json(
        { error: "데이터 조회 중 오류가 발생했습니다" },
        { status: 500 },
      );
    }

    const row = data?.[0];
    if (!row) {
      return NextResponse.json(null);
    }

    return NextResponse.json({ ...row.data, date: row.date });
  } catch (e) {
    console.error("[API /scenarios] Unexpected error:", e);
    return NextResponse.json(
      { error: "서버 오류가 발생했습니다" },
      { status: 500 },
    );
  }
}
