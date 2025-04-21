import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { projects } from '@/data/projectData/projectData';
import { Github } from 'lucide-react';
import TypingText from '../ui/loading-text';
import { Skeleton } from '../ui/skeleton';

interface ProjectImage {
  src: string;
  alt: string;
}

interface Project {
  id: string;
  highlight: boolean;
  title: string;
  url: string;
  image: ProjectImage;
  description: string;
  keyFeatures: string[];
  implementationDetails: string;
  technologies: string[];
}

export default function ProjectsSection() {
  const highlightedProjects = projects.filter(project => project.highlight);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadedProjects, setLoadedProjects] = useState<Project[]>([]);
  
  const handleProjectClick = (project: Project) => {
    setSelectedProject(project);
    setIsDialogOpen(true);
  };

  useEffect(() => {
    // Check if there are highlighted projects
    if (highlightedProjects.length === 0) {
      console.warn('No highlighted projects found. Please mark some projects with highlight: true');
    }
    
    // Shorter loading delay for better user experience
    const timer = setTimeout(() => {
      setIsLoading(false);
      setLoadedProjects(highlightedProjects);
    }, 800);
    
    return () => clearTimeout(timer);
  }, [highlightedProjects]);

  return (
    <section id="projects" className="py-20 px-4">
      <div className="container max-w-6xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold mb-4 text-center">Featured Projects</h2>
        <div className="grid md:grid-cols-2 gap-8">
          {isLoading ? (
            // Skeleton loading UI
            Array(4).fill(0).map((_, index) => (
              <Card key={index} className="overflow-hidden shadow-md pt-0 animate-fadeIn">
                <div className="aspect-video relative bg-accent/30 animate-pulse"></div>
                <CardHeader>
                  <div className="h-6 w-3/4 bg-accent/30 rounded-md animate-pulse"></div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="h-4 bg-accent/30 rounded-md animate-pulse"></div>
                    <div className="h-4 bg-accent/30 rounded-md animate-pulse"></div>
                    <div className="h-4 w-2/3 bg-accent/30 rounded-md animate-pulse"></div>
                  </div>
                  <div className="flex gap-2 mt-4">
                    <div className="h-6 w-16 bg-accent/30 rounded-full animate-pulse"></div>
                    <div className="h-6 w-16 bg-accent/30 rounded-full animate-pulse"></div>
                    <div className="h-6 w-16 bg-accent/30 rounded-full animate-pulse"></div>
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between items-center">
                  <div className="h-5 w-24 bg-accent/30 rounded-md animate-pulse"></div>
                  <div className="h-5 w-20 bg-accent/30 rounded-md animate-pulse"></div>
                </CardFooter>
              </Card>
            ))
          ) : (
            loadedProjects.map((project) => (
              <Card 
                key={project.id} 
                className="overflow-hidden shadow-md hover:shadow-lg hover:scale-[101%] active:scale-99 pt-0 transition-all duration-300 cursor-pointer animate-fadeIn"
                style={{ animationDelay: `${(parseInt(project.id.slice(-1)) % 4) * 150}ms` }}
                onClick={() => handleProjectClick(project)}
              >
                <div className="aspect-video relative overflow-hidden bg-muted group">
                  <Image 
                    src={project.image.src} 
                    alt={project.title}
                    fill
                    className="object-cover transition-transform duration-700 group-hover:scale-110"
                  />
                </div>
                <CardHeader>
                  <TypingText 
                    text={project.title}
                    className="text-lg font-semibold"
                    noTyping={true}
                  />
                </CardHeader>
                <CardContent>
                  <TypingText 
                    text={project.description}
                    className="text-muted-foreground mb-4"
                    noTyping={true}
                  />
                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.technologies.slice(0, 3).map((tech, index) => (
                      <span key={index} className="px-2 py-1 bg-accent/50 rounded text-xs hover:bg-accent hover:scale-105 transition-all duration-300">{tech}</span>
                    ))}
                    {project.technologies.length > 3 && (
                      <span className="px-2 py-1 bg-accent/50 rounded text-xs">+{project.technologies.length - 3} more</span>
                    )}
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between items-center">
                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      handleProjectClick(project);
                    }} 
                    className="text-sm font-medium hover:underline hover:text-primary transition-colors duration-300 bg-transparent border-none p-0 cursor-pointer relative after:content-[''] after:absolute after:w-0 after:h-0.5 after:bg-primary after:left-0 after:-bottom-0.5 after:transition-all hover:after:w-full"
                  >
                    View Details →
                  </button>
                  <a 
                    href="/projects" 
                    onClick={(e) => {
                      e.stopPropagation();
                      // Navigate to projects page with request access form
                    }}
                    className="flex items-center gap-1 text-sm font-medium hover:underline hover:text-primary transition-colors duration-300 group"
                  >
                    <Github size={16} className="transition-transform duration-300 group-hover:rotate-12" />
                    <span>Request Access</span>
                  </a>
                </CardFooter>
              </Card>
            ))
          )}
        </div>
        <div className="mt-12 text-center">
          <Link 
            href="/projects" 
            className="rounded-full border border-foreground/20 px-6 py-3 font-medium bg-background dark:bg-white dark:text-black hover:bg-accent dark:hover:bg-gray-100 transition-colors inline-block"
          >
            View All Projects
          </Link>
        </div>
      </div>

      {/* Project Details Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-5xl w-[95vw] md:w-[90vw] lg:w-[85vw]">
          <div className="max-h-[75vh] overflow-y-auto pr-2 [scrollbar-width:thin] [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300/40 dark:[&::-webkit-scrollbar-thumb]:bg-gray-600/40">
          {selectedProject && (
            <>
              <DialogHeader>
                <DialogTitle className="text-2xl font-bold">{selectedProject.title}</DialogTitle>
              </DialogHeader>
              
              <div className="mt-4 aspect-video relative overflow-hidden bg-muted rounded-md">
                <Image 
                  src={selectedProject.image.src} 
                  alt={selectedProject.title}
                  fill
                  className="object-cover"
                />
              </div>
              
              <div className="mt-6">
                <div className="flex justify-between items-center mb-2">
                  <h3 className="text-lg font-semibold">Description</h3>
                  <Link 
                    href="/projects#github-access-form" 
                    className="flex items-center gap-1 text-sm font-medium hover:underline"
                  >
                    <Github size={16} />
                    <span>Request Access</span>
                  </Link>
                </div>
                <p className="text-muted-foreground text-sm md:text-base leading-relaxed">{selectedProject.description}</p>
              </div>
              
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-2">Key Features</h3>
                <ul className="list-disc pl-5 space-y-1">
                  {selectedProject.keyFeatures.map((feature: string, index: number) => (
                    <li key={index} className="text-muted-foreground text-sm md:text-base">{feature}</li>
                  ))}
                </ul>
              </div>
              
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-2">Implementation Details</h3>
                <p className="text-muted-foreground text-sm md:text-base leading-relaxed">{selectedProject.implementationDetails}</p>
              </div>
              
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-2">Technologies</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedProject.technologies.map((tech: string, index: number) => (
                    <span key={index} className="px-2 py-1 bg-accent/50 rounded text-xs">{tech}</span>
                  ))}
                </div>
              </div>
            </>
          )}
          </div>
        </DialogContent>
      </Dialog>
    </section>
  );
}
