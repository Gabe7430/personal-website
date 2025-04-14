import React, { useEffect, useState } from 'react';
// Change to deploy
interface ScatteredImage {
  src: string;
  alt: string;
  position: {
    top: string;
    left: string;
    transform: string;
  };
  size: string;
  animationDelay: string;
}

export default function HomeSection() {
  const [animationComplete, setAnimationComplete] = useState(false);
  
  useEffect(() => {
    // Start animation immediately
    setAnimationComplete(true);
  }, []);
  
  const images: ScatteredImage[] = [
    {
      src: '/home-imgs/billie.png',
      alt: 'Billie',
      position: {
        top: '15%',
        left: '10%',
        transform: 'rotate(-15deg)',
      },
      size: 'h-50 w-50',
      animationDelay: '0.2s',
    },
    {
      src: '/home-imgs/mario.png',
      alt: 'Mario',
      position: {
        top: '65%',
        left: '15%',
        transform: 'rotate(-10deg)',
      },
      size: 'h-70 w-50',
      animationDelay: '0.5s',
    },
    {
      src: '/home-imgs/phoenix.jpeg',
      alt: 'Phoenix',
      position: {
        top: '25%',
        left: '75%',
        transform: 'rotate(10deg)',
      },
      size: 'h-50 w-85',
      animationDelay: '0.8s',
    },
    {
      src: '/home-imgs/hoovertower.png',
      alt: 'Hoover Tower',
      position: {
        top: '70%',
        left: '80%',
        transform: 'rotate(8deg)',
      },
      size: 'h-50 w-50',
      animationDelay: '1.1s',
    },
    {
      src: '/home-imgs/chilaquiles.png',
      alt: 'Chilaquiles',
      position: {
        top: '80%',
        left: '45%',
        transform: 'rotate(-10deg)',
      },
      size: 'h-50 w-50',
      animationDelay: '1.4s',
    },
    {
      src: '/home-imgs/santacruz_tree.png',
      alt: 'Santa Cruz Tree',
      position: {
        top: '10%',
        left: '47%',
        transform: '',
      },
      size: 'h-50 w-50',
      animationDelay: '1.7s',
    },
  ];

  return (
    <section id="home" className="min-h-screen flex flex-col justify-center items-center py-20 px-4 relative overflow-hidden bg-white">
      {/* Scattered images with circular animation */}
      {images.map((img, index) => (
        <div
          key={index}
          className={`absolute z-10 transition-all duration-[1.5s] ease-in-out ${animationComplete ? '' : 'opacity-0'}`}
          style={{
            top: img.position.top,
            left: img.position.left,
            transform: img.position.transform,
            transitionDelay: img.animationDelay,
            opacity: animationComplete ? 1 : 0,
          }}
        >
          <div 
            className={`${img.size} transition-all duration-500 ease-in-out 
              hover:scale-110 cursor-pointer opacity-100`}
            style={{ transitionDelay: img.animationDelay }}
          >
            <img
              src={img.src}
              alt={img.alt}
              className="w-full h-full object-cover rounded-md
                transition-all duration-500 
                hover:brightness-110 hover:contrast-110"
            />
          </div>
        </div>
      ))}
      
      {/* Main content */}
      <div className="container max-w-4xl mx-auto text-center flex flex-col items-center z-20 relative">
        <img 
          src="/images/home-photo.JPG" 
          alt="Profile" 
          className="rounded-full h-50 w-50 object-cover mb-7 shadow-lg" 
        />
        <h1 className="text-5xl md:text-6xl font-bold mb-6 text-gray-900">Gabe SantaCruz</h1>
        <p className="text-xl md:text-2xl mb-8 text-gray-600">Machine Learning Engineer & Full Stack Developer</p>
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
