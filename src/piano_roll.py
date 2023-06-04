import sys
import numpy as np
from glob import glob
import os
from pathlib import Path
from piano_transcription_inference import PianoTranscription
from utils_midi import TargetProcessor, read_midi


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def transcribe_midi(filename, audio_array_16, device='cuda', data_path=f'matej/cibulicka/dataset',
                    out_path='matej/cibulicka/dataset/midi',
                    model_path='model/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth', save_dict=True):
    # MIDI Transcriptor # device: 'cuda' | 'cpu'
    if not os.path.exists(f'{out_path}/{filename}.mid'):
        with HiddenPrints():
            try:
                transcriptor = PianoTranscription(device=device, checkpoint_path=model_path)
            except:
                transcriptor = PianoTranscription(device='cpu', checkpoint_path=model_path)

        midi_dict = transcriptor.transcribe(audio_array_16, f'{out_path}/{filename}.mid')
        midi_dict = midi_dict['est_note_events']
        if save_dict:
            np.save(f'{data_path}/midi_dict/{filename}.npy', midi_dict)


# needs the midi_dict
def get_onsets_dict(input_dict=None, user='matej', session='cibulicka', filename=None, merge=None, save=True):
    note_events = np.load(input_dict, allow_pickle=True)
    # note_events = midi_dict['est_note_events']
    onsets = []
    for note_event in note_events:
        onsets.append(note_event['onset_time'])
    onsets_sorted = sorted(onsets, key=float)

    if merge:
        do = True
        merge_onsets = onsets_sorted.copy()
        threshold = 0.07
        index = 0
        while do is True:
            for i, onset in enumerate(merge_onsets[index:]):
                try:
                    if merge_onsets[(i + index) + 1] - onset <= threshold:
                        merge_onsets.remove(merge_onsets[(i + index) + 1])
                        index = i + index
                        break
                    else:
                        continue
                except:
                    do = False
        onsets_sorted = merge_onsets
    if save:
        np.savetxt(f'../{user}/{session}/dataset/onsets/{filename}.txt', onsets_sorted, fmt='%.4f')
    return onsets_sorted


# extremely slow, needs the transcribed midi file but no midi_dict
def get_midi_onsets(filename=None, audio_array_16=None, midi_path=None,
                    min_note=0, max_note=None, sr=16000, save=True):
    audio_seconds = audio_array_16.shape[0] / sr
    midi_dict = read_midi(midi_path)

    target_processor = TargetProcessor(segment_seconds=audio_seconds,
                                       frames_per_second=100, begin_note=21,
                                       classes_num=88)

    (target_dict, note_events, pedal_events, list_of_onsets) = target_processor.process(
        start_time=0,
        midi_events_time=midi_dict['midi_event_time'],
        midi_events=midi_dict['midi_event'])

    note_onsets_list = []

    for d in note_events:
        temp_list = [d[i] for i in d]
        if max_note:
            if min_note <= temp_list[0] <= max_note:
                note_onsets_list.append(temp_list[1])
        else:
            if temp_list[0] >= min_note:
                note_onsets_list.append(temp_list[1])
    if save:
        if not filename:
            filename = Path(midi_path).stem
        np.savetxt(f'../{user}/{session}/dataset/onsets/{filename}.txt', note_onsets_list, fmt='%.4f')
    return note_onsets_list


def get_piano_rolls(user, session):
    data_path = f'../{user}/{session}/dataset'
    audio_paths_16 = glob(f'{data_path}/audio_arrays_16/*')
    midi_dir = f'{data_path}/midi'
    onset_dir = f'{data_path}/onsets'

    if not os.path.exists(midi_dir):
        os.makedirs(midi_dir)
    if not os.path.exists(onset_dir):
        os.makedirs(onset_dir)
    if not os.path.exists(f'{data_path}/midi_dict'):
        os.makedirs(f'{data_path}/midi_dict')

    for audio_path in audio_paths_16:
        audio_array = np.load(audio_path)
        filename = Path(audio_path).stem
        transcribe_midi(filename, audio_array, out_path=midi_dir,
                        model_path=r'../model/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth',
                        data_path=data_path)
        # get_midi_onsets(filename=filename, audio_array_16, midi_path=f'{midi_dir}/{filename}.mid')
        get_onsets_dict(input_dict=f'{data_path}/midi_dict/{filename}.npy', user=user, session=session,
                        filename=filename, merge=True)


if __name__ == '__main__':

    from config import *

    print(f'---processing {os.path.basename(__file__)}---')
    for session in sessions:
        get_piano_rolls(user, session)

    print('\ndone')
