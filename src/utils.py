import os
import numpy as np
import subprocess as sp
from scipy.interpolate import interp1d
from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor


def ffmpeg_load_audio(filepath: str,
                      sr: int,
                      mono: bool = True,
                      normalize: bool = True,
                      in_type=np.int16,
                      out_type=np.float32,
                      devnull=open(os.devnull, 'w')):
    """
    Ffmpeg function for the non .wav audio files.

    Args:
        filepath:
        sr:
        mono:
        normalize:
        in_type:
        out_type:
        devnull:

    Returns:
        audio (np.ndarray):
        sr (int):
    """

    channels = 1 if mono else 2
    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int16: 's16le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[in_type]
    command = [
        'ffmpeg',
        '-i', filepath,
        '-f', format_string,
        '-acodec', 'pcm_' + format_string,
        '-ar', str(sr),
        '-ac', str(channels),
        '-']
    p = sp.Popen(command, stdout=sp.PIPE, stderr=devnull, bufsize=4096, shell=True)
    bytes_per_sample = np.dtype(in_type).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sr  # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.frombuffer(raw, dtype=in_type).astype(out_type)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()
    if audio.size == 0:
        return audio, sr
    if issubclass(out_type, np.floating):
        if normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak
        elif issubclass(in_type, np.integer):
            audio /= np.iinfo(in_type).max
    return audio, sr


def resample_signal(x_in: np.ndarray,
                    sr_in: int = 100,
                    sr_out: int = 50,
                    norm: bool = True):
    """
    Resample given signal with input sampling rate (sr_in) to output sampling rate (sr_out).

    Args:
        x_in:
        sr_in:
        sr_out:
        norm:

    Returns:
        x_out: output resampled signal
    """

    t_coef_in = np.arange(x_in.shape[0]) / sr_in
    time_in_max_sec = t_coef_in[-1]
    time_max_sec = time_in_max_sec
    n_out = int(np.ceil(time_max_sec * sr_out))
    t_coef_out = np.arange(n_out) / sr_out
    if t_coef_out[-1] > time_in_max_sec:
        x_in = np.append(x_in, [0])
        t_coef_in = np.append(t_coef_in, [t_coef_out[-1]])
    x_out = interp1d(t_coef_in, x_in, kind='linear')(t_coef_out)
    if norm:
        x_max = max(x_out)
        if x_max > 0:
            x_out = x_out / max(x_out)
    return x_out


class PreProcessor(SequentialProcessor):
    """
    Preprocessor the audio recording to specific time-frequency represenation for the beat trracking model.
    Originally from madmom package.
    """

    def __init__(self, frame_size=1024, num_bands=12, log=np.log, add=1e-6, fps=50, sr=22050):
        # resample to a fixed sample rate in order to get always the same number of filter bins
        sig = SignalProcessor(num_channels=1, sample_rate=sr)
        # split audio signal in overlapping frames
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps)
        # compute STFT
        stft = ShortTimeFourierTransformProcessor()
        # filter the magnitudes
        filt = FilteredSpectrogramProcessor(num_bands=num_bands)
        # scale them logarithmically
        spec = LogarithmicSpectrogramProcessor(log=log, add=add)
        # instantiate a SequentialProcessor
        super(PreProcessor, self).__init__((sig, frames, stft, filt, spec, np.array))
        # safe fps as attribute (needed for quantization of events)
        self.fps = fps


