import struct
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from pipecat.frames.frames import ErrorFrame, MetricsFrame, TTSAudioRawFrame
from pipecat.metrics.metrics import ProcessingMetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

from processor.wav2arkit_blendshape import (
    Wav2ArkitBlendshapeModel,
    Wav2ArkitBlendshapeProcessor,
)


def _audio_frame(
    samples: list[int],
    *,
    sample_rate: int = 16000,
    num_channels: int = 1,
) -> TTSAudioRawFrame:
    audio = struct.pack(f"<{len(samples)}h", *samples)
    return TTSAudioRawFrame(
        audio=audio,
        sample_rate=sample_rate,
        num_channels=num_channels,
    )


class Wav2ArkitBlendshapeModelTests(unittest.TestCase):
    def test_predict_converts_single_batch_output_to_lists(self):
        session = Mock()
        session.run.return_value = [
            np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
        ]
        model = Wav2ArkitBlendshapeModel(session=session)
        audio_input = np.array([[0.0, 0.5]], dtype=np.float32)

        result = model.predict(audio_input)

        session.run.assert_called_once()
        output_names, inputs = session.run.call_args.args
        self.assertIsNone(output_names)
        self.assertIs(inputs["audio_waveform"], audio_input)
        np.testing.assert_allclose(result, [[0.1, 0.2], [0.3, 0.4]])

    def test_predict_rejects_unexpected_output_shape(self):
        session = Mock()
        session.run.return_value = [np.zeros((2, 3, 4), dtype=np.float32)]
        model = Wav2ArkitBlendshapeModel(session=session)

        with self.assertRaisesRegex(
            ValueError,
            r"unexpected wav2arkit output shape: \(2, 3, 4\)",
        ):
            model.predict(np.zeros((1, 10), dtype=np.float32))


class Wav2ArkitAudioInputTests(unittest.TestCase):
    def test_mono_pcm_is_normalized_to_float32(self):
        frame = _audio_frame([-32768, 0, 16384, 32767])

        audio_input = Wav2ArkitBlendshapeProcessor._audio_input(frame)

        self.assertEqual(audio_input.shape, (1, 4))
        self.assertEqual(audio_input.dtype, np.float32)
        np.testing.assert_allclose(
            audio_input,
            [[-1.0, 0.0, 0.5, 32767 / 32768]],
        )

    def test_stereo_pcm_is_mixed_down_to_mono(self):
        frame = _audio_frame(
            [-32768, 32767, 16384, 0],
            num_channels=2,
        )

        audio_input = Wav2ArkitBlendshapeProcessor._audio_input(frame)

        np.testing.assert_allclose(
            audio_input,
            [[(-32768 + 32767) / 2 / 32768, 0.25]],
        )

    @patch("processor.wav2arkit_blendshape.resample_poly")
    def test_audio_is_resampled_to_16khz(self, resample_poly):
        resample_poly.return_value = np.array([0.25, 0.5], dtype=np.float64)
        frame = _audio_frame([0, 16384], sample_rate=24000)

        audio_input = Wav2ArkitBlendshapeProcessor._audio_input(frame)

        source_audio, up, down = resample_poly.call_args.args
        np.testing.assert_allclose(source_audio, [0.0, 0.5])
        self.assertEqual((up, down), (2, 3))
        self.assertEqual(audio_input.dtype, np.float32)
        np.testing.assert_allclose(audio_input, [[0.25, 0.5]])

    def test_invalid_audio_metadata_and_incomplete_pcm_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample_rate"):
            Wav2ArkitBlendshapeProcessor._audio_input(
                _audio_frame([0], sample_rate=0)
            )

        invalid_channels = _audio_frame([0])
        invalid_channels.num_channels = 0
        with self.assertRaisesRegex(ValueError, "num_channels"):
            Wav2ArkitBlendshapeProcessor._audio_input(
                invalid_channels
            )

        with self.assertRaisesRegex(ValueError, "complete PCM samples"):
            Wav2ArkitBlendshapeProcessor._audio_input(
                _audio_frame([0, 1, 2], num_channels=2)
            )


class Wav2ArkitBlendshapeProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_downstream_audio_is_forwarded_before_blendshapes(self):
        model = Mock()
        model.predict.return_value = [[0.1, 0.2], [0.3, 0.4]]
        processor = Wav2ArkitBlendshapeProcessor(model=model)
        processor.push_frame = AsyncMock()
        frame = _audio_frame([0] * 1600)
        frame.pts = 1234

        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(processor.push_frame.await_count, 2)
        first_call, second_call = processor.push_frame.await_args_list
        self.assertIs(first_call.args[0], frame)
        self.assertIs(first_call.args[1], FrameDirection.DOWNSTREAM)

        message = second_call.args[0]
        self.assertIsInstance(message, RTVIServerMessageFrame)
        self.assertEqual(
            message.data,
            {
                "type": "blendshapes",
                "data": [[0.1, 0.2], [0.3, 0.4]],
                "pts": 1234,
            },
        )
        self.assertIs(second_call.args[1], FrameDirection.DOWNSTREAM)

        audio_input = model.predict.call_args.args[0]
        np.testing.assert_allclose(audio_input, np.zeros((1, 1600)))

    async def test_upstream_audio_is_only_forwarded(self):
        model = Mock()
        processor = Wav2ArkitBlendshapeProcessor(model=model)
        processor.push_frame = AsyncMock()
        frame = _audio_frame([0])

        await processor.process_frame(frame, FrameDirection.UPSTREAM)

        processor.push_frame.assert_awaited_once_with(
            frame,
            FrameDirection.UPSTREAM,
        )
        model.predict.assert_not_called()

    async def test_inference_failure_does_not_drop_audio(self):
        model = Mock()
        model.predict.side_effect = RuntimeError("inference failed")
        processor = Wav2ArkitBlendshapeProcessor(model=model)
        processor.push_frame = AsyncMock()
        frame = _audio_frame([0])

        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(processor.push_frame.await_count, 2)
        audio_call, error_call = processor.push_frame.await_args_list
        self.assertEqual(
            audio_call.args,
            (frame, FrameDirection.DOWNSTREAM),
        )
        self.assertIsInstance(error_call.args[0], ErrorFrame)
        self.assertEqual(
            error_call.args[0].error,
            "Wav2Arkit processing failed",
        )
        self.assertIsInstance(error_call.args[0].exception, RuntimeError)
        self.assertIs(error_call.args[1], FrameDirection.UPSTREAM)

    def test_processor_supports_pipecat_metrics(self):
        processor = Wav2ArkitBlendshapeProcessor(model=Mock())

        self.assertTrue(processor.can_generate_metrics())

    async def test_processor_emits_pipecat_processing_metrics(self):
        model = Mock()
        model.predict.return_value = [[0.1, 0.2]]
        processor = Wav2ArkitBlendshapeProcessor(model=model)
        processor._enable_metrics = True
        processor.push_frame = AsyncMock()

        await processor.process_frame(
            _audio_frame([0]),
            FrameDirection.DOWNSTREAM,
        )

        metrics_frame = processor.push_frame.await_args_list[-1].args[0]
        self.assertIsInstance(metrics_frame, MetricsFrame)
        self.assertEqual(len(metrics_frame.data), 1)
        self.assertIsInstance(metrics_frame.data[0], ProcessingMetricsData)
        self.assertEqual(metrics_frame.data[0].processor, processor.name)

    def test_model_and_session_cannot_both_be_injected(self):
        with self.assertRaisesRegex(
            ValueError,
            "model and session cannot both be provided",
        ):
            Wav2ArkitBlendshapeProcessor(
                model=Mock(),
                session=Mock(),
            )


if __name__ == "__main__":
    unittest.main()
