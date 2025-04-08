import React from 'react';
import Link from 'next/link';

export default function Footer() {
  return (
    <footer className="py-8 px-4">
      <div className="container mx-auto max-w-6xl">
        <div className="bg-gray-100 rounded-lg px-6 py-8 mx-4 md:mx-8 lg:mx-12">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0">
              <p className="text-sm text-muted-foreground">© {new Date().getFullYear()} Gabe SantaCruz. All rights reserved.</p>
            </div>
            <div className="flex gap-6">
              <Link href="/#home" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Home</Link>
              <Link href="/#about" className="text-sm text-muted-foreground hover:text-foreground transition-colors">About</Link>
              <Link href="/#projects" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Projects</Link>
              <Link href="/#contact" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Contact</Link>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
