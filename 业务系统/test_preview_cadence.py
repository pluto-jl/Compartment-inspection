import unittest
from preview_cadence import PreviewCadence


class PreviewCadenceTests(unittest.TestCase):
    def test_first_and_final_latest_frames_retained(self):
        clock = [0.]
        out = []
        cadence = PreviewCadence(out.append, clock=lambda: clock[0])
        for index in range(10):
            clock[0] = index * .01
            cadence.submit(index)
        self.assertEqual(out, [0])
        cadence.flush()
        self.assertEqual(out, [0, 9])
        cadence.flush()
        self.assertEqual(out, [0, 9])

    def test_slow_input_keeps_each_available_frame(self):
        clock = [0.]
        out = []
        cadence = PreviewCadence(out.append, clock=lambda: clock[0])
        for index in range(4):
            clock[0] = index * .5
            cadence.submit(index)
        cadence.flush()
        self.assertEqual(out, list(range(4)))

    def test_failed_encoding_does_not_discard_pending_frame(self):
        def fail(frame):
            raise ValueError('encode failed')
        cadence = PreviewCadence(fail)
        frame = object()
        with self.assertRaises(ValueError):
            cadence.submit(frame)
        self.assertIs(cadence.pending, frame)
        self.assertEqual(cadence.encoded, 0)


if __name__ == '__main__':
    unittest.main()
