import React, { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Geist, Geist_Mono } from "next/font/google";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { projects } from '@/data/projectData/projectData';
import GitHubAccessForm from '@/components/GitHubAccessForm';
import ContactSection from '@/components/sections/ContactSection';
import { Github } from 'lucide-react';

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

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export default function Projects() {
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [activeFilters, setActiveFilters] = useState<string[]>([]);
  const [filteredProjects, setFilteredProjects] = useState<Project[]>(projects);
  
  // Define filter categories
  const filterCategories = [
    "Machine Learning",
    "Computer Vision",
    "Deep Learning",
    "Python",
    "C/C++",
    "NumPy",
    "PyTorch",
    "TensorFlow",
    "Reinforcement Learning",
    "Natural Language Processing"
  ];
  
  // Handle filter toggle
  const toggleFilter = (filter: string) => {
    setActiveFilters(prev => {
      if (prev.includes(filter)) {
        return prev.filter(f => f !== filter);
      } else {
        return [...prev, filter];
      }
    });
  };
  
  // Filter projects based on active filters
  useEffect(() => {
    if (activeFilters.length === 0) {
      setFilteredProjects(projects);
    } else {
      const filtered = projects.filter(project => {
        return activeFilters.some(filter => {
          // Check if any of the project's technologies include the filter
          // Use case-insensitive matching and handle special cases
          return project.technologies.some(tech => {
            const lowerTech = tech.toLowerCase();
            const lowerFilter = filter.toLowerCase();
            
            // Handle special cases
            if (lowerFilter === "c/c++" && (lowerTech.includes("c++") || lowerTech === "c")) {
              return true;
            }
            
            // Handle PyTorch/TensorFlow as separate filters
            if ((lowerFilter === "pytorch" && lowerTech.includes("pytorch")) ||
                (lowerFilter === "tensorflow" && lowerTech.includes("tensorflow"))) {
              return true;
            }
            
            // General case
            return lowerTech.includes(lowerFilter);
          });
        });
      });
      setFilteredProjects(filtered);
    }
  }, [activeFilters]);
  
  const handleProjectClick = (project: Project) => {
    setSelectedProject(project);
    setIsDialogOpen(true);
  };
  return (
    <div className={`${geistSans.variable} ${geistMono.variable} min-h-screen font-sans`}>
      
      <main className="py-20 px-4">
        <div className="container max-w-6xl mx-auto">
          <div className="mb-16">
            <h1 className="text-4xl md:text-5xl font-bold mb-6 text-center">Portfolio</h1>
            <p className="text-center text-muted-foreground max-w-2xl mx-auto">
              A collection of my work in machine learning, computer vision, reinforcement learning, and overall Artificial Intelligence! Click on a project to learn more!
            </p>
          </div>
          
          <div className="mb-8">
            <h2 className="text-xl font-semibold mb-4 text-center">Filter by</h2>
            <div className="flex flex-wrap justify-center gap-2 max-w-3xl mx-auto">
              {filterCategories.map((filter) => (
                <button
                  key={filter}
                  onClick={() => toggleFilter(filter)}
                  className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all duration-300 ${
                    activeFilters.includes(filter)
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-accent/50 hover:bg-accent/70 text-foreground'
                  }`}
                >
                  {filter}
                </button>
              ))}
              {activeFilters.length > 0 && (
                <button
                  onClick={() => setActiveFilters([])}
                  className="px-3 py-1.5 rounded-full text-sm font-medium bg-destructive/10 hover:bg-destructive/20 text-destructive transition-all duration-300"
                >
                  Clear Filters
                </button>
              )}
            </div>
            {activeFilters.length > 0 && (
              <p className="text-center text-sm text-muted-foreground mt-2">
                Showing {filteredProjects.length} of {projects.length} projects
              </p>
            )}
          </div>
          
          <div id="github-access-form" className="mb-16 max-w-md mx-auto">
            <h2 className="text-2xl font-semibold mb-4 text-center">Request GitHub Access</h2>
            <GitHubAccessForm />
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {filteredProjects.map((project) => (
              <div 
                key={project.id} 
                className={`border hover:scale-[101%] active:scale-99 rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-all duration-300 cursor-pointer ${
                  project.highlight ? 'border-[#185761] border-2 shadow-md shadow-primary/10' : ''
                }`}
                onClick={() => handleProjectClick(project)}
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
                      <span className="bg-primary text-primary-foreground text-xs px-2 py-1 rounded-full font-medium shadow-sm">
                        Featured
                      </span>
                    </div>
                  )}
                </div>
                <div className="p-6">
                  <h2 className="text-xl font-semibold mb-2">{project.title}</h2>
                  <p className="text-muted-foreground mb-4 text-sm line-clamp-3">{project.description}</p>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.technologies.slice(0, 3).map((tech, index) => (
                      <span key={index} className="px-2 py-1 bg-accent/50 rounded text-xs">{tech}</span>
                    ))}
                    {project.technologies.length > 3 && (
                      <span className="px-2 py-1 bg-accent/50 rounded text-xs">+{project.technologies.length - 3} more</span>
                    )}
                  </div>
                  <div className="flex justify-between items-center">
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        handleProjectClick(project);
                      }} 
                      className="text-sm font-medium hover:underline bg-transparent border-none p-0 cursor-pointer"
                    >
                      View Details →
                    </button>
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        // Scroll to the GitHub access form
                        document.querySelector('#github-access-form')?.scrollIntoView({ behavior: 'smooth' });
                      }}
                      className="flex items-center gap-1 text-sm font-medium hover:underline cursor-pointer bg-transparent border-0"
                    >
                      <Github size={16} />
                      <span>Request Access</span>
                    </button>
                  </div>
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
      
      {/* Contact Section */}
      <ContactSection />

      {/* Project Details Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
  <DialogContent className="max-w-7xl w-[98vw] md:w-[95vw] lg:w-[90vw]">
    <div className="max-h-[75vh] overflow-y-auto pr-2 [scrollbar-width:thin] [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300/40 dark:[&::-webkit-scrollbar-thumb]:bg-gray-600/40">
      {selectedProject && (
        <>
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold">{selectedProject.title}</DialogTitle>
          </DialogHeader>

          <div className="mt-4 aspect-video relative rounded-md">
            <Image 
              src={selectedProject.image.src} 
              alt={selectedProject.title}
              fill
              className="object-contain"
            />
          </div>

          <div className="mt-6">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-lg font-semibold">Description</h3>
              <button 
                onClick={() => {
                  document.querySelector('#github-access-form')?.scrollIntoView({ behavior: 'smooth' });
                  setIsDialogOpen(false);
                }}
                className="flex items-center gap-1 text-sm font-medium hover:underline cursor-pointer bg-transparent border-0"
              >
                <Github size={16} />
                <span>Request Access</span>
              </button>
            </div>
            <p className="text-muted-foreground text-sm md:text-base leading-relaxed">
              {selectedProject.description}
            </p>
          </div>

          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-2">Key Features</h3>
            <ul className="list-disc pl-5 space-y-1">
              {selectedProject.keyFeatures.map((feature: string, index: number) => (
                <li key={index} className="text-muted-foreground text-sm md:text-base">
                  {feature}
                </li>
              ))}
            </ul>
          </div>

          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-2">Implementation Details</h3>
            <p className="text-muted-foreground text-sm md:text-base leading-relaxed">
              {selectedProject.implementationDetails}
            </p>
          </div>

          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-2">Technologies</h3>
            <div className="flex flex-wrap gap-2">
              {selectedProject.technologies.map((tech: string, index: number) => (
                <span key={index} className="px-2 py-1 bg-accent/50 rounded text-xs">
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  </DialogContent>
    </Dialog>
    </div>
  );
}
