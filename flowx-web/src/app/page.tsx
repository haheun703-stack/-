import MorningBriefing from "@/components/MorningBriefing";
import SignalScoreboard from "@/components/SignalScoreboard";

export default function Home() {
  // TODO: 사용자 인증 후 isPaid 결정
  const isPaid = false;

  return (
    <div className="space-y-10">
      {/* 모닝 브리핑 */}
      <section>
        <MorningBriefing isPaid={isPaid} />
      </section>

      {/* 구분선 */}
      <div className="border-t border-gray-800" />

      {/* 성적표 + 시그널 */}
      <section>
        <h2 className="text-white text-xl font-bold mb-4">📊 시그널 성적표</h2>
        <SignalScoreboard isPaid={isPaid} />
      </section>
    </div>
  );
}
