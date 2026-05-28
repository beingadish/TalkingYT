import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Talking YouTube",
  description: "Chat with one or more YouTube transcripts."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

