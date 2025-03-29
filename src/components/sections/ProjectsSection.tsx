import React from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { projects } from '@/data/projectsData';

export default function ProjectsSection() {
  const highlightedProjects = projects.filter(project => project.highlight);

  return (
    <section id="projects" className="py-20 px-4">
      <div className="container max-w-6xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold mb-12 text-center">Featured Projects</h2>
        <div className="grid md:grid-cols-2 gap-8">
          {highlightedProjects.map((project) => (
            <Card key={project.id} className="overflow-hidden shadow-md hover:shadow-lg hover:scale-[101%] active:scale-99 pt-0 transition-shadow">
              <div className="aspect-video relative overflow-hidden bg-muted">
                <Image 
                  src={project.image.src} 
                  alt={project.title}
                  fill
                  className="object-cover"
                />
              </div>
              <CardHeader>
                <CardTitle>{project.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">{project.shortDescription}</p>
                <div className="flex flex-wrap gap-2 mb-4">
                  {project.technologies.slice(0, 3).map((tech, index) => (
                    <span key={index} className="px-2 py-1 bg-accent/50 rounded text-xs">{tech}</span>
                  ))}
                  {project.technologies.length > 3 && (
                    <span className="px-2 py-1 bg-accent/50 rounded text-xs">+{project.technologies.length - 3} more</span>
                  )}
                </div>
              </CardContent>
              <CardFooter>
                <Link href={`/projects/${project.id}`} className="text-sm font-medium hover:underline">
                  View Details →
                </Link>
              </CardFooter>
            </Card>
          ))}
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
    </section>
  );
}
