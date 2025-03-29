import React from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Geist, Geist_Mono } from "next/font/google";
import { projects } from '@/data/projectsData';

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export default function Projects() {
  return (
    <div className={`${geistSans.variable} ${geistMono.variable} min-h-screen font-sans`}>
      
      <main className="py-20 px-4">
        <div className="container max-w-6xl mx-auto">
          <div className="mb-16">
            <h1 className="text-4xl md:text-5xl font-bold mb-6 text-center">My Projects</h1>
            <p className="text-center text-muted-foreground max-w-2xl mx-auto">
              A collection of my work in machine learning, computer vision, reinforcement learning, and web development.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {projects.map((project) => (
              <div 
                key={project.id} 
                className={`border hover:scale-[101%] active:scale-99 rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-shadow ${
                  project.highlight ? 'ring-2 ring-primary/20' : ''
                }`}
              >
                <div className="aspect-video relative overflow-hidden bg-muted">
                  <Image 
                    src={project.image.src} 
                    alt={project.title}
                    fill
                    className="object-cover"
                  />
                  {project.highlight && (
                    <div className="absolute top-2 right-2">
                      <span className="bg-primary text-primary-foreground text-xs px-2 py-1 rounded-full">
                        Featured
                      </span>
                    </div>
                  )}
                </div>
                <div className="p-6">
                  <h2 className="text-xl font-semibold mb-2">{project.title}</h2>
                  <p className="text-muted-foreground mb-4 text-sm line-clamp-3">{project.shortDescription}</p>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.technologies.slice(0, 3).map((tech, index) => (
                      <span key={index} className="px-2 py-1 bg-accent/50 rounded text-xs">{tech}</span>
                    ))}
                    {project.technologies.length > 3 && (
                      <span className="px-2 py-1 bg-accent/50 rounded text-xs">+{project.technologies.length - 3} more</span>
                    )}
                  </div>
                  <Link href={`/projects/${project.id}`} className="text-sm font-medium hover:underline">
                    View Details →
                  </Link>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-16 text-center">
            <Link 
              href="/" 
              className="rounded-full border border-foreground/20 px-6 py-3 font-medium bg-background dark:bg-white dark:text-black hover:bg-accent dark:hover:bg-gray-100 transition-colors inline-block"
            >
              Back to Home
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
