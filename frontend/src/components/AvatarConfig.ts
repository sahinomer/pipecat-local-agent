export const AVATAR_MODEL_URL = '/avatar.glb';

export const AVATAR_SETTINGS = {
  DATA_FPS: 30,
  CLAY_MODE: false,
  SHADOWS: true,

  // ActorCore uses texture alpha for details such as hair and eyelashes.
  ALPHA_TEST: 0.35,

  // Higher damping follows incoming expressions more quickly.
  DAMPING: 0.4,

  // Limits the change applied to a blendshape during one render frame.
  MAX_VELOCITY: 0.12,

  // Filters small changes from the incoming animation stream.
  NOISE_THRESHOLD: 0.01,
} as const;

export const FRAME_TIME = 1 / AVATAR_SETTINGS.DATA_FPS;

export const JAW_SETTINGS = {
  BONE_NAME: 'JawRoot',
  AXIS: 'z',
  FACTOR: 0.333,
} as const;

// Procedural blinking is layered on top of any blink values sent by the server.
// Durations are in seconds.
export const BLINK_SETTINGS = {
  ENABLED: true,
  MIN_INTERVAL: 2.5,
  MAX_INTERVAL: 6,
  CLOSE_DURATION: 0.09,
  HOLD_DURATION: 0.035,
  OPEN_DURATION: 0.14,
  STRENGTH: 1,
} as const;

// Maps the API's ARKit names to the morph target names exported in the GLB.
export const MORPH_MAPPING: Record<string, string> = {
  "browInnerUp": "A01_Brow_Inner_Up",
  "browDownLeft": "A02_Brow_Down_Left",
  "browDownRight": "A03_Brow_Down_Right",
  "browOuterUpLeft": "A04_Brow_Outer_Up_Left",
  "browOuterUpRight": "A05_Brow_Outer_Up_Right",
  "eyeLookUpLeft": "A06_Eye_Look_Up_Left",
  "eyeLookUpRight": "A07_Eye_Look_Up_Right",
  "eyeLookDownLeft": "A08_Eye_Look_Down_Left",
  "eyeLookDownRight": "A09_Eye_Look_Down_Right",
  "eyeLookOutLeft": "A10_Eye_Look_Out_Left",
  "eyeLookInLeft": "A11_Eye_Look_In_Left",
  "eyeLookInRight": "A12_Eye_Look_In_Right",
  "eyeLookOutRight": "A13_Eye_Look_Out_Right",
  "eyeBlinkLeft": "A14_Eye_Blink_Left",
  "eyeBlinkRight": "A15_Eye_Blink_Right",
  "eyeSquintLeft": "A16_Eye_Squint_Left",
  "eyeSquintRight": "A17_Eye_Squint_Right",
  "eyeWideLeft": "A18_Eye_Wide_Left",
  "eyeWideRight": "A19_Eye_Wide_Right",
  "cheekPuff": "A20_Cheek_Puff",
  "cheekSquintLeft": "A21_Cheek_Squint_Left",
  "cheekSquintRight": "A22_Cheek_Squint_Right",
  "noseSneerLeft": "A23_Nose_Sneer_Left",
  "noseSneerRight": "A24_Nose_Sneer_Right",
  "jawOpen": "A25_Jaw_Open",
  "jawForward": "A26_Jaw_Forward",
  "jawLeft": "A27_Jaw_Left",
  "jawRight": "A28_Jaw_Right",
  "mouthFunnel": "A29_Mouth_Funnel",
  "mouthPucker": "A30_Mouth_Pucker",
  "mouthLeft": "A31_Mouth_Left",
  "mouthRight": "A32_Mouth_Right",
  "mouthRollUpper": "A33_Mouth_Roll_Upper",
  "mouthRollLower": "A34_Mouth_Roll_Lower",
  "mouthShrugUpper": "A35_Mouth_Shrug_Upper",
  "mouthShrugLower": "A36_Mouth_Shrug_Lower",
  "mouthClose": "A37_Mouth_Close",
  "mouthSmileLeft": "A38_Mouth_Smile_Left",
  "mouthSmileRight": "A39_Mouth_Smile_Right",
  "mouthFrownLeft": "A40_Mouth_Frown_Left",
  "mouthFrownRight": "A41_Mouth_Frown_Right",
  "mouthDimpleLeft": "A42_Mouth_Dimple_Left",
  "mouthDimpleRight": "A43_Mouth_Dimple_Right",
  "mouthUpperUpLeft": "A44_Mouth_Upper_Up_Left",
  "mouthUpperUpRight": "A45_Mouth_Upper_Up_Right",
  "mouthLowerDownLeft": "A46_Mouth_Lower_Down_Left",
  "mouthLowerDownRight": "A47_Mouth_Lower_Down_Right",
  "mouthPressLeft": "A48_Mouth_Press_Left",
  "mouthPressRight": "A49_Mouth_Press_Right",
  "mouthStretchLeft": "A50_Mouth_Stretch_Left",
  "mouthStretchRight": "A51_Mouth_Stretch_Right",
};


// This order is part of the server protocol; change it only when the API changes.
export const ARKIT_ORDER = [
  "browDownLeft",
  "browDownRight",
  "browInnerUp",
  "browOuterUpLeft",
  "browOuterUpRight",
  "cheekPuff",
  "cheekSquintLeft",
  "cheekSquintRight",
  "eyeBlinkLeft",
  "eyeBlinkRight",
  "eyeLookDownLeft",
  "eyeLookDownRight",
  "eyeLookInLeft",
  "eyeLookInRight",
  "eyeLookOutLeft",
  "eyeLookOutRight",
  "eyeLookUpLeft",
  "eyeLookUpRight",
  "eyeSquintLeft",
  "eyeSquintRight",
  "eyeWideLeft",
  "eyeWideRight",
  "jawForward",
  "jawLeft",
  "jawOpen",
  "jawRight",
  "mouthClose",
  "mouthDimpleLeft",
  "mouthDimpleRight",
  "mouthFrownLeft",
  "mouthFrownRight",
  "mouthFunnel",
  "mouthLeft",
  "mouthLowerDownLeft",
  "mouthLowerDownRight",
  "mouthPressLeft",
  "mouthPressRight",
  "mouthPucker",
  "mouthRight",
  "mouthRollLower",
  "mouthRollUpper",
  "mouthShrugLower",
  "mouthShrugUpper",
  "mouthSmileLeft",
  "mouthSmileRight",
  "mouthStretchLeft",
  "mouthStretchRight",
  "mouthUpperUpLeft",
  "mouthUpperUpRight",
  "noseSneerLeft",
  "noseSneerRight",
  "tongueOut",
] as const;
