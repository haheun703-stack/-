import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import SearchBar from "@/components/SearchBar";
import MobileNav from "@/components/MobileNav";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FLOWX | AI \uC8FC\uC2DD \uC2DC\uADF8\uB110",
  description: "\uD140\uD2B8 + \uB2E8\uD0C0 + AI \uAD50\uCC28\uAC80\uC99D \uC2DC\uADF8\uB110 \uD50C\uB7AB\uD3FC",
};

const NAV_LINKS = [
  { href: "/", label: "\uB300\uC2DC\uBCF4\uB4DC" },
  { href: "/quant", label: "\uD140\uD2B8" },
  { href: "/scenarios", label: "\uC2DC\uB098\uB9AC\uC624" },
  { href: "/signals", label: "\uC2DC\uADF8\uB110" },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-black min-h-screen`}
      >
        {/* 네비게이션 */}
        <nav className="border-b border-gray-800 bg-black/95 backdrop-blur sticky top-0 z-50">
          <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2">
              <span className="text-xl font-bold text-gray-100">FLOWX</span>
              <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
                BETA
              </span>
            </Link>

            {/* 데스크톱 네비 */}
            <div className="hidden md:flex items-center gap-4">
              {NAV_LINKS.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="text-gray-400 hover:text-gray-100 text-sm transition-colors"
                >
                  {link.label}
                </Link>
              ))}
              <SearchBar />
            </div>

            {/* 모바일 메뉴 버튼 */}
            <div className="md:hidden">
              <MobileNav links={NAV_LINKS} />
            </div>
          </div>
        </nav>

        <main className="max-w-5xl mx-auto px-4 py-6">{children}</main>

        {/* 푸터 */}
        <footer className="border-t border-gray-800 mt-12 py-6 text-center text-gray-600 text-xs">
          <p>
            FLOWX | AI {"\uAE30\uBC18"} {"\uC8FC\uC2DD"} {"\uC2DC\uADF8\uB110"} &middot;{" "}
            {"\uD22C\uC790"} {"\uD310\uB2E8\uC740"} {"\uBCF8\uC778"} {"\uCC45\uC784"}
          </p>
        </footer>
      </body>
    </html>
  );
}
