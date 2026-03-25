import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const botType = searchParams.get("bot_type") || "QUANT";
  const status = searchParams.get("status"); // OPEN, CLOSED, STOPPED or null(all)
  const limit = parseInt(searchParams.get("limit") || "20", 10);

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
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data || []);
}
