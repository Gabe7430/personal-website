import React from 'react';

export default function HomeSection() {
  return (
    <section id="home" className="min-h-screen flex flex-col justify-center items-center py-20 px-4">
      <div className="container max-w-4xl mx-auto text-center flex flex-col items-center">
        <img src="https://i.pinimg.com/1200x/c0/92/1c/c0921c5f03376ccb36af0be21495595b.jpg" alt="Profile"  className="rounded-full h-32 w-32 object-cover mb-7" />
        <h1 className="text-5xl md:text-6xl font-bold mb-6">Gabe SantaCruz</h1>
        <p className="text-xl md:text-2xl mb-8 text-muted-foreground">Machine Learning Engineer & Full Stack Developer</p>
        <div className="flex gap-4 justify-center flex-wrap">
          <a 
            href="#projects" 
            className="rounded-full bg-foreground text-background px-6 py-3 font-medium hover:bg-foreground/90 transition-colors"
          >
            View My Work
          </a>
          <a 
            href="#contact" 
            className="rounded-full border border-foreground/20 px-6 py-3 font-medium hover:bg-accent transition-colors"
          >
            Contact Me
          </a>
        </div>
      </div>
    </section>
  );
}
