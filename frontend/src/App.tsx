// src/App.tsx
import { useState, useEffect } from 'react';
import LandingScreen from './LandingScreen';
import RecourseWorkspace from './RecourseWorkspace';

function App() {
  const [view, setView] = useState<'landing' | 'workspace'>('landing');
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setVisible(false);
    const id = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(id);
  }, [view]);

  return (
    <div className={`transition-opacity duration-300 ${visible ? 'opacity-100' : 'opacity-0'}`}>
      {view === 'landing' ? (
        <LandingScreen onEnter={() => setView('workspace')} />
      ) : (
        <RecourseWorkspace />
      )}
    </div>
  );
}

export default App;