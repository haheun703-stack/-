import SectorFireDashboard from "@/components/SectorFireDashboard";

export const metadata = {
  title: "섹터발화 | FLOWX",
  description: "FIRE + Structure = Composite 섹터 종합 등급",
};

export default function SectorFirePage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">섹터발화</h1>
        <p className="text-gray-500 text-sm mt-1">
          FIRE × 0.6 + Structure × 0.4 = Composite 종합 등급
        </p>
      </div>
      <SectorFireDashboard />
    </div>
  );
}
