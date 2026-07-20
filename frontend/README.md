# 3D Avatar Client

React frontend for a real-time Pipecat voice agent with a blendshape-driven 3D avatar.

## How it works

Pipecat provides the real-time connection between the browser and the voice-agent server. While the agent speaks, the server also streams ordered blendshape frames that describe its facial expressions.

```text
Pipecat server
      |
      v
ordered blendshape frames
      |
      v
queue, timing, and smoothing
      |
      v
semantic-to-model morph mapping
      |
      v
Three.js GLB avatar
```

The animation pipeline buffers the incoming frames, applies them at the expected cadence, smooths rapid changes, and maps each semantic expression to a morph target in the loaded model. Three.js, through React Three Fiber, applies the resulting weights and renders the avatar in real time.

The current client expects an ARKit-style set of 52 blendshape values. The avatar itself is model-agnostic: any GLB model with suitable facial blendshapes can be used by configuring the morph-target mapping.

## Setup

Use npm with a Vite-supported Node.js release: Node.js `20.19` or newer in the 20.x line, or Node.js `22.12` or newer. From the repository root, run:

```bash
cd frontend
npm ci
npm run dev
```

The development server prints its local URL, normally `http://localhost:5173`.

Set the Pipecat WebSocket endpoint when it differs from the default `ws://localhost:7860/ws`:

```env
VITE_BACKEND_URL=ws://localhost:7860/ws
```

## Required avatar model

No avatar model is included. Copy a compatible GLB model to:

```text
public/avatar.glb
```

If the filename or location changes, update `AVATAR_MODEL_URL` in `src/components/AvatarConfig.ts`.

The model should be a GLB containing facial morph targets (blendshapes). A jaw bone is optional. The current configuration uses an ARKit-style expression ordering and maps those semantic names to the morph-target names defined in the loaded model.

The client was tested with a free ActorCore avatar, but any compatible GLB model can be used.

## ARKit data and blendshape mapping

The server sends messages in this form:

```json
{
  "type": "blendshapes",
  "data": [[0.0, 0.12, 0.0]]
}
```

`data` is a batch of animation frames. Each frame must follow `ARKIT_ORDER` in `src/components/AvatarConfig.ts`; the current client consumes frames at 30 FPS. The example above is shortened; production frames should contain one value for every expression in that order.

`MORPH_MAPPING` separates the protocol from the model:

```ts
export const MORPH_MAPPING = {
  eyeBlinkLeft: 'Model_Blink_Left',
  jawOpen: 'Model_Jaw_Open',
  mouthSmileLeft: 'Model_Smile_Left',
  // ...
};
```

- The **key** is the semantic expression sent by the server.
- The **value** is the exact morph-target name stored in the GLB.
- Missing mappings are skipped, so partial expression sets can still be used.

### Replacing the model

1. Put the new GLB in `public/` and update `AVATAR_MODEL_URL` if necessary.
2. Update the values in `MORPH_MAPPING` to match the new GLB exactly.
3. If the rig uses a jaw bone, update `JAW_SETTINGS.BONE_NAME`, `AXIS`, and `FACTOR`. Adjust the avatar position or camera in `Avatar.tsx` and `App.tsx` if the new model has a different scale or origin.
4. Run the app and test the new model's animation with the Pipecat server.

Do not reorder `ARKIT_ORDER` for a model-only change. It defines the client/server wire protocol. To use a non-ARKit expression schema, change the ordered expression list and server output together, then map those semantic expressions to the new model's morph targets.

## Production build

```bash
npm run build
```

The generated `dist/` directory is ignored by Git.

## License

Code and documentation original to this repository are covered by the [MIT License](../LICENSE).
