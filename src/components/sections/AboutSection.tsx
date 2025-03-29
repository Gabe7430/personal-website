import React from 'react';

export default function AboutSection() {
  return (
    <section id="about" className="py-20 px-4 bg-muted/50">
      <div className="container max-w-4xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold mb-12 text-center">About Me</h2>
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <h3 className="text-2xl font-semibold mb-4">Machine Learning Engineer</h3>
            <p className="mb-4 text-foreground/80 dark:text-foreground/90">
              I specialize in developing machine learning solutions for complex problems, with expertise in computer vision, 
              reinforcement learning, and natural language processing.
            </p>
            <p className="mb-4 text-foreground/80 dark:text-foreground/90">
              With a strong background in both theoretical foundations and practical implementations, 
              I build systems that deliver real-world value across various domains.
            </p>
            <div className="flex flex-wrap gap-2 mt-6">
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium">PyTorch</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium">TensorFlow</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium">Computer Vision</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium">NLP</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium">Reinforcement Learning</span>
            </div>
          </div>
          <div>
            <h3 className="text-2xl font-semibold mb-4">Full Stack Developer</h3>
            <p className="mb-4 text-foreground/80 dark:text-foreground/90">
              I build modern web applications using React, Next.js, and other cutting-edge technologies, 
              with a focus on creating intuitive and responsive user experiences.
            </p>
            <p className="mb-4 text-foreground/80 dark:text-foreground/90">
              My approach combines technical expertise with a keen eye for design, 
              ensuring that the applications I develop are both functional and visually appealing.
            </p>
            <div className="flex flex-wrap gap-2 mt-6">
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium">React</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium">Next.js</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium">TypeScript</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium">Node.js</span>
              <span className="px-3 py-1 bg-accent dark:bg-primary/20 text-accent-foreground dark:text-primary border border-accent/20 dark:border-primary/20 rounded-full text-sm font-medium">Tailwind CSS</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
