import { useEffect, useState, useRef } from 'react';
import { useInView } from '@/hooks/useInView';

interface TypingTextProps {
  text: string;
  speed?: number; // milliseconds per character
  className?: string;
  onComplete?: () => void;
  startOnView?: boolean; // Whether to start typing when element is in view
  threshold?: number; // Intersection observer threshold
}

export default function TypingText({ 
  text, 
  speed = 30, 
  className = "", 
  onComplete,
  startOnView = true,
  threshold = 0.1
}: TypingTextProps) {
  const [displayedText, setDisplayedText] = useState('');
  const [hasStarted, setHasStarted] = useState(!startOnView);
  const [ref, isInView] = useInView<HTMLParagraphElement>({ threshold, once: true });
  
  useEffect(() => {
    // Only start typing if hasStarted is true
    if (!hasStarted) return;
    
    // Reset text when starting
    setDisplayedText('');
    
    // Create a copy of the text to avoid any reference issues
    const textToType = String(text);
    let typedText = '';
    let position = 0;
    
    const interval = setInterval(() => {
      if (position < textToType.length) {
        // Add one character at a time
        typedText += textToType.charAt(position);
        setDisplayedText(typedText);
        position++;
      } else {
        // We've reached the end of the text
        clearInterval(interval);
        if (onComplete) onComplete();
      }
    }, speed);

    // Clean up interval on unmount or when dependencies change
    return () => clearInterval(interval);
  }, [text, speed, onComplete, hasStarted]);
  
  // Start typing when element comes into view
  useEffect(() => {
    if (isInView && !hasStarted && startOnView) {
      setHasStarted(true);
    }
  }, [isInView, hasStarted, startOnView]);

  return (
    <p 
      ref={ref} 
      className={`whitespace-pre-wrap ${className}`}
    >
      {displayedText}
    </p>
  );
}