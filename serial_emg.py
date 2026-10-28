"""Realtime 8-channel EMG input from an ESP32 serial port.

Expected default wire format (ASCII, one sample per line):
    ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n
The ESP32 source rate is assumed to be 1000 Hz.  Samples are filtered,
resampled to 689.06 Hz, and normalized to match the historical .npy path.

Install dependency:
    pip install pyserial
"""

import re
import time
from collections import deque

import numpy as np
import scipy.signal

try:
    import serial
except ImportError as exc:  # pragma: no cover
    serial = None
    _SERIAL_IMPORT_ERROR = exc
else:
    _SERIAL_IMPORT_ERROR = None


class StreamingEMGPreprocessor:
    """Causal streaming approximation of the original offline preprocessing.

    Original offline path:
      60-Hz notch harmonics -> 2-Hz high-pass -> 1000->689.06 Hz
      -> /20 -> 50*tanh(x/50)

    Note: the original code uses filtfilt (zero-phase).  A true live stream has
    no future samples, so this implementation uses stateful causal IIR filters.
    This is appropriate for live operation, but the phase response is not
    numerically identical to the old offline preprocessing.
    """

    def __init__(self, source_rate=1000.0, target_rate=689.06,
                 num_channels=8, input_scale=1.0):
        self.source_rate = float(source_rate)
        self.target_rate = float(target_rate)
        self.num_channels = int(num_channels)
        self.input_scale = float(input_scale)

        if self.source_rate <= 0 or self.target_rate <= 0:
            raise ValueError("sample rates must be > 0")

        # Match read_emg.py: notch 60,120,...,420 Hz (valid under 500-Hz Nyquist)
        self.notch_filters = []
        for harmonic in range(1, 8):
            freq = 60.0 * harmonic
            if freq >= self.source_rate / 2:
                continue
            b, a = scipy.signal.iirnotch(freq, 30, self.source_rate)
            self.notch_filters.append((b, a))

        self.hp_b, self.hp_a = scipy.signal.butter(
            3, 2, 'highpass', fs=self.source_rate
        )

        self.notch_zi = None
        self.hp_zi = None
        self.input_index = 0
        self.output_index = 0
        self.prev_filtered = None

    def reset(self):
        self.notch_zi = None
        self.hp_zi = None
        self.input_index = 0
        self.output_index = 0
        self.prev_filtered = None

    @staticmethod
    def _init_zi(b, a, first_row):
        base = scipy.signal.lfilter_zi(b, a)[:, None]
        return base * first_row[None, :]

    def _filter(self, x):
        y = np.asarray(x, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != self.num_channels:
            raise ValueError(
                f"expected (N,{self.num_channels}) input, got {y.shape}"
            )
        if len(y) == 0:
            return y

        y = y * self.input_scale

        if self.notch_zi is None:
            self.notch_zi = [
                self._init_zi(b, a, y[0]) for b, a in self.notch_filters
            ]

        for i, (b, a) in enumerate(self.notch_filters):
            y, self.notch_zi[i] = scipy.signal.lfilter(
                b, a, y, axis=0, zi=self.notch_zi[i]
            )

        if self.hp_zi is None:
            self.hp_zi = self._init_zi(self.hp_b, self.hp_a, y[0])

        y, self.hp_zi = scipy.signal.lfilter(
            self.hp_b, self.hp_a, y, axis=0, zi=self.hp_zi
        )
        return y

    def _resample_streaming(self, filtered):
        """Linear interpolation matching the old np.interp approach."""
        if len(filtered) == 0:
            return np.empty((0, self.num_channels), dtype=np.float32)

        start = self.input_index
        end = start + len(filtered) - 1
        step = self.source_rate / self.target_rate

        if self.prev_filtered is None:
            ext = filtered
            x_coords = np.arange(start, end + 1, dtype=np.float64)
        else:
            ext = np.vstack([self.prev_filtered[None, :], filtered])
            x_coords = np.arange(start - 1, end + 1, dtype=np.float64)

        positions = []
        while True:
            pos = self.output_index * step
            if pos > end + 1e-12:
                break
            if pos >= x_coords[0] - 1e-12:
                positions.append(pos)
            self.output_index += 1

        self.input_index += len(filtered)
        self.prev_filtered = filtered[-1].copy()

        if not positions:
            return np.empty((0, self.num_channels), dtype=np.float32)

        positions = np.asarray(positions, dtype=np.float64)
        out = np.empty((len(positions), self.num_channels), dtype=np.float64)
        for ch in range(self.num_channels):
            out[:, ch] = np.interp(positions, x_coords, ext[:, ch])

        # Match the historical normalization in read_emg.py / GUI loader.
        out = out / 20.0
        out = 50.0 * np.tanh(out / 50.0)
        return out.astype(np.float32, copy=False)

    def process(self, raw_samples):
        return self._resample_streaming(self._filter(raw_samples))


class SerialEMGSource:
    """Iterator yielding processed EMG chunks with shape (N, 8)."""

    def __init__(self, port, baudrate=921600, source_rate=1000.0,
                 target_rate=689.06, output_chunk_samples=34,
                 timeout=0.1, input_scale=1.0, num_channels=8):
        if serial is None:
            raise RuntimeError(
                "pyserial is not installed. Run: pip install pyserial"
            ) from _SERIAL_IMPORT_ERROR

        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.num_channels = int(num_channels)
        self.output_chunk_samples = int(output_chunk_samples)
        self.preprocessor = StreamingEMGPreprocessor(
            source_rate=source_rate,
            target_rate=target_rate,
            num_channels=num_channels,
            input_scale=input_scale,
        )

        self.ser = None
        self.running = False
        self._processed = deque()
        self.bad_lines = 0
        self.total_lines = 0

    def open(self):
        if self.ser is not None and self.ser.is_open:
            return
        self.ser = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout,
        )
        # Drop stale bytes collected before this process started.
        self.ser.reset_input_buffer()
        self.running = True

    def close(self):
        self.running = False
        if self.ser is not None:
            try:
                if self.ser.is_open:
                    self.ser.close()
            finally:
                self.ser = None

    def stop(self):
        self.close()

    def _parse_line(self, raw_line):
        self.total_lines += 1
        try:
            line = raw_line.decode('utf-8', errors='ignore').strip()
        except AttributeError:
            line = str(raw_line).strip()
        if not line:
            return None

        # Accept "EMG:1,2,..." as well as plain CSV/space-separated values.
        if ':' in line:
            prefix, rest = line.split(':', 1)
            if not re.search(r'[0-9.+-]', prefix):
                line = rest.strip()

        tokens = [t for t in re.split(r'[,;\s]+', line) if t]
        try:
            values = [float(t) for t in tokens]
        except ValueError:
            self.bad_lines += 1
            return None

        if len(values) < self.num_channels:
            self.bad_lines += 1
            return None

        # If a timestamp/counter is prepended, taking the last 8 values keeps
        # the channel payload intact.  If your timestamp is appended instead,
        # change this slice to values[:8].
        values = values[-self.num_channels:]
        return np.asarray(values, dtype=np.float32)

    def _read_raw_batch(self, max_samples=64):
        rows = []
        while self.running and len(rows) < max_samples:
            raw = self.ser.readline()
            if not raw:
                if rows:
                    break
                continue
            row = self._parse_line(raw)
            if row is not None:
                rows.append(row)
        if not rows:
            return np.empty((0, self.num_channels), dtype=np.float32)
        return np.stack(rows, axis=0)

    def __iter__(self):
        self.open()
        pending = np.empty((0, self.num_channels), dtype=np.float32)

        try:
            while self.running:
                raw_batch = self._read_raw_batch()
                if len(raw_batch) == 0:
                    continue

                processed = self.preprocessor.process(raw_batch)
                if len(processed) == 0:
                    continue

                if len(pending) == 0:
                    pending = processed
                else:
                    pending = np.vstack([pending, processed])

                while len(pending) >= self.output_chunk_samples:
                    chunk = pending[:self.output_chunk_samples]
                    pending = pending[self.output_chunk_samples:]
                    yield chunk
        finally:
            self.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ESP32 8-ch EMG serial test')
    parser.add_argument('--port', required=True)
    parser.add_argument('--baud', type=int, default=921600)
    parser.add_argument('--rate', type=float, default=1000.0)
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()

    src = SerialEMGSource(
        port=args.port,
        baudrate=args.baud,
        source_rate=args.rate,
        input_scale=args.scale,
        output_chunk_samples=34,
    )
    print(f'Connected to {args.port} @ {args.baud}. Ctrl+C to stop.')
    try:
        for chunk in src:
            print(
                f'shape={chunk.shape}  '
                f'min={chunk.min():.3f}  max={chunk.max():.3f}  '
                f'rms={np.sqrt(np.mean(chunk**2)):.3f}'
            )
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
