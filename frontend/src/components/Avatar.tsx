import { useEffect, useMemo, useRef } from 'react';
import type { MutableRefObject } from 'react';
import { useGLTF, ContactShadows } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

import { 
  ARKIT_ORDER,
  AVATAR_SETTINGS, 
  AVATAR_MODEL_URL,
  BLINK_SETTINGS,
  FRAME_TIME, 
  JAW_SETTINGS,
  MORPH_MAPPING,
} from './AvatarConfig';

const BLENDSHAPE_COUNT = ARKIT_ORDER.length;
const JAW_OPEN_INDEX = ARKIT_ORDER.indexOf('jawOpen');
const LEFT_BLINK_INDEX = ARKIT_ORDER.indexOf('eyeBlinkLeft');
const RIGHT_BLINK_INDEX = ARKIT_ORDER.indexOf('eyeBlinkRight');

interface AvatarProps {
  animationQueue: MutableRefObject<number[][]>;
}

export function Avatar({ animationQueue }: AvatarProps) {
  const { scene } = useGLTF(AVATAR_MODEL_URL);
  
  const jawBoneRef = useRef<THREE.Object3D | null>(null);
  const initialJawRotation = useRef<THREE.Euler | null>(null);

  const timeAccumulator = useRef(0);
  const currentValues = useRef(new Float32Array(BLENDSHAPE_COUNT));
  const targetValues = useRef(new Float32Array(BLENDSHAPE_COUNT));
  const blinkTimer = useRef(0);
  const blinkProgress = useRef<number | null>(null);
  const nextBlinkDelay = useRef<number>(BLINK_SETTINGS.MIN_INTERVAL);
  const proceduralBlinkValue = useRef(0);

  useEffect(() => {
    scene.traverse((child) => {
      const mesh = child as THREE.Mesh;

      if (mesh.isMesh) {
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        if (AVATAR_SETTINGS.CLAY_MODE) {
          mesh.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color('#eeeeee'),
            roughness: 0.5,
          });
        } else {
          const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];

          materials.forEach((material) => {
            // ActorCore's single BLEND + DoubleSide material renders the back
            // of lips and eyelids through the front of the face. Alpha cutout
            // keeps hair/eyelashes transparent while restoring depth sorting.
            material.side = THREE.FrontSide;
            material.transparent = false;
            material.alphaTest = AVATAR_SETTINGS.ALPHA_TEST;
            material.depthTest = true;
            material.depthWrite = true;

            if (material instanceof THREE.MeshStandardMaterial && material.map) {
              material.envMapIntensity = 1.0;
              material.roughness = 0.6;
            }

            material.needsUpdate = true;
          });
        }
      }

      const name = child.name.toLowerCase();
      const targetName = JAW_SETTINGS.BONE_NAME.toLowerCase();
      const isJawCandidate = (child.type === 'Bone' || child.type === 'Object3D')
        && (name === targetName || name.includes('jaw') || name.includes('chin'));

      // Prefer the configured name, but support common names from other exporters.
      if (isJawCandidate && (name === targetName || !jawBoneRef.current)) {
        jawBoneRef.current = child;
        initialJawRotation.current = child.rotation.clone();
      }
    });
  }, [scene]);

  const morphTargets = useMemo(() => {
    const targets: { mesh: THREE.Mesh; indexMap: number[] }[] = [];

    scene.traverse((child) => {
      const mesh = child as THREE.Mesh;
      const dictionary = mesh.morphTargetDictionary;

      if (!mesh.isMesh || !dictionary) return;

      const indexMap = ARKIT_ORDER.map((name) => {
        const mappedName = MORPH_MAPPING[name];
        return mappedName !== undefined && dictionary[mappedName] !== undefined
          ? dictionary[mappedName]
          : -1;
      });

      if (indexMap.some((index) => index !== -1)) {
        targets.push({ mesh, indexMap });
      }
    });

    return targets;
  }, [scene]);

  useFrame((_, delta) => {
    timeAccumulator.current += delta;

    if (BLINK_SETTINGS.ENABLED) {
      if (blinkProgress.current === null) {
        blinkTimer.current += delta;

        if (blinkTimer.current >= nextBlinkDelay.current) {
          blinkProgress.current = 0;
          blinkTimer.current = 0;
        }
      } else {
        blinkProgress.current += delta;

        const closeEnd = BLINK_SETTINGS.CLOSE_DURATION;
        const holdEnd = closeEnd + BLINK_SETTINGS.HOLD_DURATION;
        const blinkEnd = holdEnd + BLINK_SETTINGS.OPEN_DURATION;
        const elapsed = blinkProgress.current;

        if (elapsed < closeEnd) {
          const progress = elapsed / closeEnd;
          proceduralBlinkValue.current = progress * progress * (3 - 2 * progress);
        } else if (elapsed < holdEnd) {
          proceduralBlinkValue.current = 1;
        } else if (elapsed < blinkEnd) {
          const progress = (elapsed - holdEnd) / BLINK_SETTINGS.OPEN_DURATION;
          const easedProgress = progress * progress * (3 - 2 * progress);
          proceduralBlinkValue.current = 1 - easedProgress;
        } else {
          proceduralBlinkValue.current = 0;
          blinkProgress.current = null;
          nextBlinkDelay.current = THREE.MathUtils.randFloat(
            BLINK_SETTINGS.MIN_INTERVAL,
            BLINK_SETTINGS.MAX_INTERVAL,
          );
        }
      }
    }
    
    // Consume frames at the server's cadence even when rendering at a higher FPS.
    while (timeAccumulator.current >= FRAME_TIME) {
      const rawInput = animationQueue.current.shift();

      if (rawInput) {
        ARKIT_ORDER.forEach((_, index) => {
          const value = rawInput[index] ?? 0;
          targetValues.current[index] = value < AVATAR_SETTINGS.NOISE_THRESHOLD ? 0 : value;
        });
      } else {
        targetValues.current.fill(0);
      }

      timeAccumulator.current -= FRAME_TIME;
    }

    ARKIT_ORDER.forEach((_, index) => {
      const current = currentValues.current[index];
      const target = targetValues.current[index];
      const step = THREE.MathUtils.clamp(
        (target - current) * AVATAR_SETTINGS.DAMPING,
        -AVATAR_SETTINGS.MAX_VELOCITY,
        AVATAR_SETTINGS.MAX_VELOCITY,
      );

      currentValues.current[index] = current + step;
    });

    // Layer procedural blinking over server data so either source can close the eyes.
    morphTargets.forEach(({ mesh, indexMap }) => {
      const influences = mesh.morphTargetInfluences;
      if (!influences) return;

      indexMap.forEach((meshIndex, index) => {
        if (meshIndex === -1) return;

        const isBlink = index === LEFT_BLINK_INDEX || index === RIGHT_BLINK_INDEX;
        const blinkValue = proceduralBlinkValue.current * BLINK_SETTINGS.STRENGTH;
        influences[meshIndex] = isBlink
          ? Math.max(currentValues.current[index], blinkValue)
          : currentValues.current[index];
      });
    });

    if (jawBoneRef.current && initialJawRotation.current && JAW_OPEN_INDEX !== -1) {
      const jawOpenAmount = currentValues.current[JAW_OPEN_INDEX];

      jawBoneRef.current.rotation.copy(initialJawRotation.current);
      jawBoneRef.current.rotation[JAW_SETTINGS.AXIS] += jawOpenAmount * JAW_SETTINGS.FACTOR;
    }
  });

  return (
    <group dispose={null} position={[0, -1.55, 0]}>
      <primitive object={scene} />
      {AVATAR_SETTINGS.SHADOWS && (
        <ContactShadows opacity={0.4} scale={10} blur={2.5} far={4} color="#000000" />
      )}
    </group>
  );
}

useGLTF.preload(AVATAR_MODEL_URL);
