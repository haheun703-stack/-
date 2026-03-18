import SignalScoreboard from "@/components/SignalScoreboard";

export default function SignalsPage() {
  const isPaid = false;

  return (
    <div>
      <h1 className="text-white text-2xl font-bold mb-6">📡 시그널 대시보드</h1>
      <SignalScoreboard isPaid={isPaid} />
    </div>
  );
}
