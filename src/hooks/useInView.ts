import { useState, useEffect, useRef, RefObject } from 'react';

interface InViewOptions {
  threshold?: number;
  rootMargin?: string;
  once?: boolean;
}

export function useInView<T extends Element>({
  threshold = 0.1,
  rootMargin = '0px',
  once = false,
}: InViewOptions = {}): [RefObject<T>, boolean] {
  // Using MutableRefObject instead of RefObject to fix type issue
  const ref = useRef<T>(null) as React.MutableRefObject<T>;
  const [isInView, setIsInView] = useState(false);

  useEffect(() => {
    if (!ref.current) return;
    
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          if (once && ref.current) {
            observer.unobserve(ref.current);
          }
        } else if (!once) {
          setIsInView(false);
        }
      },
      { threshold, rootMargin }
    );

    observer.observe(ref.current);

    return () => {
      if (ref.current) {
        observer.unobserve(ref.current);
      }
    };
  }, [threshold, rootMargin, once]);

  return [ref, isInView];
}
