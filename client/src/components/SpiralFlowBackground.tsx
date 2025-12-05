/**
 * Spiral Flow Background Component
 * 
 * Premium animated background inspired by Spiral Flow template
 * Features: Particle system, gradient waves, smooth animations
 */

import { useEffect, useRef } from 'react';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  color: string;
  opacity: number;
}

export function SpiralFlowBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animationFrameRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // Initialize particles
    const particleCount = 80;
    const colors = [
      'rgba(59, 130, 246, 0.6)',  // blue-500
      'rgba(6, 182, 212, 0.6)',   // cyan-500
      'rgba(139, 92, 246, 0.6)',  // violet-500
      'rgba(236, 72, 153, 0.6)',  // pink-500
    ];

    particlesRef.current = Array.from({ length: particleCount }, (_,__) => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      radius: Math.random() * 3 + 1,
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: Math.random() * 0.5 + 0.3,
    }));

    // Animation loop
    let time = 0;
    const animate = () => {
      time += 0.01;
      
      // Clear canvas with fade effect
      ctx.fillStyle = 'rgba(2, 6, 23, 0.05)'; // slate-950 with low opacity
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw spiral gradient waves
      const gradient = ctx.createRadialGradient(
        canvas.width / 2,
        canvas.height / 2,
        0,
        canvas.width / 2,
        canvas.height / 2,
        Math.max(canvas.width, canvas.height) / 2
      );
      gradient.addColorStop(0, 'rgba(59, 130, 246, 0.03)');
      gradient.addColorStop(0.5, 'rgba(6, 182, 212, 0.02)');
      gradient.addColorStop(1, 'rgba(2, 6, 23, 0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Update and draw particles
      particlesRef.current.forEach((particle, index) => {
        // Update position
        particle.x += particle.vx;
        particle.y += particle.vy;

        // Add spiral motion
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const dx = particle.x - centerX;
        const dy = particle.y - centerY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx);
        const spiralForce = 0.001;
        particle.vx += Math.cos(angle + Math.PI / 2) * spiralForce;
        particle.vy += Math.sin(angle + Math.PI / 2) * spiralForce;

        // Boundary check with wrap-around
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.x > canvas.width) particle.x = 0;
        if (particle.y < 0) particle.y = canvas.height;
        if (particle.y > canvas.height) particle.y = 0;

        // Pulsing opacity
        particle.opacity = 0.3 + Math.sin(time + index * 0.1) * 0.3;

        // Draw particle
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
        ctx.fillStyle = particle.color.replace(/[\d.]+\)$/, `${particle.opacity})`);
        ctx.fill();

        // Draw connections to nearby particles
        particlesRef.current.forEach((otherParticle, otherIndex) => {
          if (otherIndex <= index) return;
          
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 120) {
            ctx.beginPath();
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            const opacity = (1 - distance / 120) * 0.15;
            ctx.strokeStyle = `rgba(59, 130, 246, ${opacity})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        });
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    // Cleanup
    return () => {
      window.removeEventListener('resize', resize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 -z-10 pointer-events-none"
      style={{ background: 'linear-gradient(to bottom, #020617, #0f172a)' }}
    />
  );
}
