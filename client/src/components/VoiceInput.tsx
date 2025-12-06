/**
 * Voice Input Component
 * 
 * Web Speech API integration with waveform visualization
 * Supports multiple languages and continuous recognition
 */

import { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface VoiceInputProps {
  onTranscript: (text: string) => void;
  onError?: (error: string) => void;
  language?: string;
  continuous?: boolean;
  className?: string;
}

// Extend Window interface for Web Speech API
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
  interface SpeechRecognitionEvent extends Event {
    results: SpeechRecognitionResultList;
  }
  interface SpeechRecognitionErrorEvent extends Event {
    error: string;
  }
}

export function VoiceInput({
  onTranscript,
  onError,
  language = 'en-US',
  continuous = false,
  className,
}: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const [audioLevel, setAudioLevel] = useState(0);
  const recognitionRef = useRef<any | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Check browser support
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setIsSupported(false);
      onError?.('Speech recognition is not supported in this browser');
    }
  }, [onError]);

  // Initialize audio visualization
  const initializeAudioVisualization = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      const microphone = audioContext.createMediaStreamSource(stream);
      
      analyser.fftSize = 256;
      microphone.connect(analyser);
      
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      // Start visualization loop
      const visualize = () => {
        if (!analyserRef.current) return;

        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        
        // Calculate average audio level
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
        setAudioLevel(average / 255);

        animationFrameRef.current = requestAnimationFrame(visualize);
      };

      visualize();
    } catch (error) {
      console.error('Error initializing audio visualization:', error);
      onError?.('Microphone access denied');
    }
  };

  // Cleanup audio visualization
  const cleanupAudioVisualization = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    setAudioLevel(0);
  };

  // Start voice recognition
  const startRecording = () => {
    if (!isSupported) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.lang = language;
    recognition.continuous = continuous;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsRecording(true);
      initializeAudioVisualization();
    };

    recognition.onresult = (event: any) => {
      const transcript = Array.from(event.results)
        .map((result: any) => result[0].transcript)
        .join('');
      
      if (event.results[event.results.length - 1].isFinal) {
        onTranscript(transcript);
      }
    };

    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      onError?.(event.error);
      stopRecording();
    };

    recognition.onend = () => {
      if (continuous && isRecording) {
        // Restart if continuous mode
        recognition.start();
      } else {
        stopRecording();
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  // Stop voice recognition
  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    cleanupAudioVisualization();
    setIsRecording(false);
  };

  // Toggle recording
  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);

  if (!isSupported) {
    return (
      <Button
        variant="ghost"
        size="icon"
        disabled
        className={cn('relative', className)}
        title="Speech recognition not supported"
      >
        <MicOff className="w-5 h-5 text-gray-400" />
      </Button>
    );
  }

  return (
    <div className="relative">
      <Button
        variant={isRecording ? 'default' : 'ghost'}
        size="icon"
        onClick={toggleRecording}
        className={cn(
          'relative transition-all duration-300',
          isRecording && 'bg-red-500 hover:bg-red-600',
          className
        )}
        title={isRecording ? 'Stop recording' : 'Start voice input'}
      >
        {isRecording ? (
          <>
            <Mic className="w-5 h-5 text-white animate-pulse" />
            {/* Waveform visualization */}
            <div
              className="absolute inset-0 rounded-full border-2 border-red-400 animate-ping"
              style={{
                opacity: audioLevel,
                transform: `scale(${1 + audioLevel * 0.5})`,
              }}
            />
          </>
        ) : (
          <Mic className="w-5 h-5" />
        )}
      </Button>

      {/* Audio level indicator */}
      {isRecording && (
        <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 flex gap-0.5">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="w-1 bg-red-500 rounded-full transition-all duration-100"
              style={{
                height: `${Math.max(2, audioLevel * 20 * (1 + Math.sin(Date.now() / 100 + i)))}px`,
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// Language options for voice input
export const VOICE_LANGUAGES = [
  { code: 'en-US', name: 'English (US)' },
  { code: 'en-GB', name: 'English (UK)' },
  { code: 'nb-NO', name: 'Norwegian (Bokmål)' },
  { code: 'nn-NO', name: 'Norwegian (Nynorsk)' },
  { code: 'sv-SE', name: 'Swedish' },
  { code: 'da-DK', name: 'Danish' },
  { code: 'de-DE', name: 'German' },
  { code: 'fr-FR', name: 'French' },
  { code: 'es-ES', name: 'Spanish' },
  { code: 'it-IT', name: 'Italian' },
  { code: 'pt-PT', name: 'Portuguese' },
  { code: 'nl-NL', name: 'Dutch' },
  { code: 'pl-PL', name: 'Polish' },
  { code: 'ru-RU', name: 'Russian' },
  { code: 'ja-JP', name: 'Japanese' },
  { code: 'zh-CN', name: 'Chinese (Simplified)' },
  { code: 'ko-KR', name: 'Korean' },
  { code: 'ar-SA', name: 'Arabic' },
  { code: 'hi-IN', name: 'Hindi' },
];
