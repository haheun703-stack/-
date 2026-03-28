"use client";

import { useState } from "react";
import Link from "next/link";
import SearchBar from "./SearchBar";

interface NavLink {
  href: string;
  label: string;
}

export default function MobileNav({ links }: { links: NavLink[] }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(!open)}
        className="text-gray-400 hover:text-gray-100 p-2 transition-colors"
        aria-label={open ? "\uBA54\uB274 \uB2EB\uAE30" : "\uBA54\uB274 \uC5F4\uAE30"}
      >
        {open ? (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        )}
      </button>

      {open && (
        <div className="absolute top-full left-0 right-0 bg-black/95 backdrop-blur border-b border-gray-800 z-50">
          <div className="max-w-5xl mx-auto px-4 py-4 space-y-3">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setOpen(false)}
                className="block text-gray-400 hover:text-gray-100 text-sm py-2 transition-colors"
              >
                {link.label}
              </Link>
            ))}
            <div className="pt-2">
              <SearchBar />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
