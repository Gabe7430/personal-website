import React, { useState } from 'react';
import Image from 'next/image';
import TypingText from '../ui/loading-text';

export default function AboutSection() {
  return (
    <section id="about" className="py-20 px-4 bg-muted/50">
      <div className="container max-w-4xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold mb-12 text-center">About Me</h2>
        <div className="space-y-8">
          <div className="flex flex-col md:flex-row gap-8 items-center animate-fadeIn" style={{ animationDelay: '300ms' }}>
            <div className="md:w-1/3 flex justify-center">
              <div className="relative w-48 h-48 rounded-full overflow-hidden border-4 border-primary/20 transition-all duration-700 hover:border-primary/50 hover:shadow-lg hover:shadow-primary/20">
                <Image 
                  src="/images/about-section-photo.jpeg" 
                  alt="Profile" 
                  fill 
                  className="object-cover transition-transform duration-700 hover:scale-110"
                />
              </div>
            </div>
            <div className="md:w-2/3">
              <TypingText 
                text="I'm from the warm desert of Phoenix, Arizona and currently study Computer Science at Stanford. From my experience, I find it hard to separate my interests and background from my work and studies. Many of my projects explore areas like dreams, robotics, movies, games, and the political and economic state of the world... all of which are personal passions of mine. Feel free to reach out about any of these topics!"
                className="mb-4 text-foreground/90 leading-relaxed"
                speed={20}
                startOnView={true}
                threshold={0.2}
              />
            </div>
          </div>
          
          <div className="space-y-4 animate-fadeIn" style={{ animationDelay: '600ms' }}>
            <TypingText 
              text="My journey began in high school where I self-taught programming as part of the robotics club. Competing in FIRST competitions and advancing to regionals was an incredible experience that shaped my early technical skills. When I arrived at Stanford, I initially pursued pre-med, but after taking my first formal computer science class, I fell in love with the CS department and shifted gears."
              className="text-foreground/90 leading-relaxed mb-4"
              speed={20}
              startOnView={true}
              threshold={0.2}
            />
            
            <TypingText 
              text="I was naturally drawn to the Artificial Intelligence track and have since taken several graduate-level courses in machine learning, deep learning, reinforcement learning, natural language processing, and data mining. The projects in my portfolio showcase much of what I've been working on since my start in robotics, and I'm excited to continue this journey!"
              className="text-foreground/90 leading-relaxed"
              speed={20}
              startOnView={true}
              threshold={0.2}
            />
          </div>
          
          <div className="pt-4 animate-fadeIn" style={{ animationDelay: '900ms' }}>
            <h3 className="text-xl font-semibold mb-4">Technical Skills</h3>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium transition-all duration-300 hover:bg-primary/10 hover:scale-105 hover:shadow-sm">PyTorch</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium transition-all duration-300 hover:bg-primary/10 hover:scale-105 hover:shadow-sm">TensorFlow</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium transition-all duration-300 hover:bg-primary/10 hover:scale-105 hover:shadow-sm">Computer Vision</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium transition-all duration-300 hover:bg-primary/10 hover:scale-105 hover:shadow-sm">NLP</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium transition-all duration-300 hover:bg-primary/10 hover:scale-105 hover:shadow-sm">Reinforcement Learning</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium transition-all duration-300 hover:bg-primary/10 hover:scale-105 hover:shadow-sm">React</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium transition-all duration-300 hover:bg-primary/10 hover:scale-105 hover:shadow-sm">Next.js</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium transition-all duration-300 hover:bg-primary/10 hover:scale-105 hover:shadow-sm">TypeScript</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
