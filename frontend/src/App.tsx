import { Canvas } from '@react-three/fiber';
import { Environment, OrbitControls } from '@react-three/drei';

import { Avatar } from './components/Avatar';
import { AvatarAgentInterface } from './components/AvatarAgentInterface';
import { useAvatarAgent } from './hooks/useAvatarAgent';

function App() {
  const backendUrl = import.meta.env.VITE_BACKEND_URL || 'ws://localhost:7860/ws';

  const {
    connect,
    disconnect,
    connectionStatus,
    chatHistory,
    animationQueue,
    isMuted,
    toggleMute,
  } = useAvatarAgent(backendUrl);

  return (
    <AvatarAgentInterface
      chatHistory={chatHistory}
      connectionStatus={connectionStatus}
      isMuted={isMuted}
      onConnect={connect}
      onDisconnect={disconnect}
      onToggleMute={toggleMute}
    >
      <Canvas shadows camera={{ position: [0, 0.2, 0.5], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 5, 5]} intensity={1} />

        <Avatar animationQueue={animationQueue} />
        <OrbitControls target={[0, 0, 0]} enablePan />
        <Environment preset="city" />
      </Canvas>
    </AvatarAgentInterface>
  );
}

export default App;
