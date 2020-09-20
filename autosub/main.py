#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import wave
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from deepspeech import Model, version 
from segmentAudio import silenceRemoval
from audioProcessing import extract_audio, convert_samplerate
from writeToFile import write_to_file
from timeit import default_timer as timer


# Line count for SRT file
line_count = 0


# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_ALPHA = 0.75

# The beta hyperparameter of the CTC decoder. Word insertion bonus.
LM_BETA = 1.85

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

def sort_alphanumeric(data):
    """Sort function to sort os.listdir() alphanumerically
    Helps to process audio files sequentially after splitting 

    Args:
        data : file name
    """
    
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    
    return sorted(data, key = alphanum_key)


def ds_process_audio(ds, audio_file, file_handle):  
    """Run DeepSpeech inference on each audio file generated after silenceRemoval
    and write to file pointed by file_handle

    Args:
        ds : DeepSpeech Model
        audio_file : audio file
        file_handle : SRT file handle
    """
    
    global line_count
    fin = wave.open(audio_file, 'rb')
    fs_orig = fin.getframerate()
    desired_sample_rate = ds.sampleRate()
    
    # Check if sampling rate is required rate (16000)
    if fs_orig != desired_sample_rate:
        print("Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition".format(fs_orig, desired_sample_rate), file=sys.stderr)
        audio = convert_samplerate(audio_file, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    fin.close()
    
    # Perform inference on audio segment
    infered_text = ds.stt(audio)
    
    # File name contains start and end times in seconds. Extract that
    limits = audio_file.split("/")[-1][:-4].split("_")[-1].split("-")
    
    if len(infered_text) != 0:
        line_count += 1
        write_to_file(file_handle, infered_text, line_count, limits)


def main():
    global line_count
    print("AutoSub v0.1\n")
        
    parser = argparse.ArgumentParser(description="AutoSub v0.1")
    parser.add_argument('--model', required=True,
                        help='DeepSpeech model file')
    parser.add_argument('--scorer',
                        help='DeepSpeech scorer file')
    parser.add_argument('--file', required=True,
                        help='Input video file')
    parser.add_argument('--alphabet', required=False,
                        help='Path to the configuration file specifying the alphabet used by the network')
    parser.add_argument('--lm', nargs='?',
                        help='Path to the language model binary file')
    parser.add_argument('--trie', nargs='?',
                        help='Path to the language model trie file created with native_client/generate_trie')
    args = parser.parse_args()
    
    ds_model = args.model
    if not ds_model.endswith(".pbmm"):
        print("Invalid model file. Exiting\n")
        exit(1)
    
    # Load DeepSpeech model 
    # ds = Model(ds_model)
    print('Loading model from file {}'.format(args.model), file=sys.stderr)
    model_load_start = timer()
    # ds = Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet, BEAM_WIDTH)
    ds = Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet)
           
    if args.scorer:
        ds_scorer = args.scorer
        if not ds_scorer.endswith(".scorer"):
            print("Invalid scorer file. Running inference using only model file\n")
        else:
            ds.enableExternalScorer(ds_scorer)
    if args.lm and args.trie:
        print('Loading language model from files {} {}'.format(args.lm, args.trie), file=sys.stderr)
        lm_load_start = timer()
        ds.enableDecoderWithLM(args.alphabet, args.lm, args.trie, LM_ALPHA, LM_BETA)
        lm_load_end = timer() - lm_load_start
        print('Loaded language model in {:.3}s.'.format(lm_load_end), file=sys.stderr)


    input_file = args.file
    print("\nInput file:", input_file)
    
    base_directory = os.getcwd()
    output_directory = os.path.join(base_directory, "output")
    audio_directory = os.path.join(base_directory, "audio")
    video_file_name = input_file.split("/")[-1].split(".")[0]
    audio_file_name = os.path.join(audio_directory, video_file_name + ".wav")
    srt_file_name = os.path.join(output_directory, video_file_name + ".srt")
    
    # Extract audio from input video file
    extract_audio(input_file, audio_file_name)
    
    print("Splitting on silent parts in audio file")
    silenceRemoval(audio_file_name)
    
    # Output SRT file
    file_handle = open(srt_file_name, "a+")
    
    print("\nRunning inference:")
    
    for file in tqdm(sort_alphanumeric(os.listdir(audio_directory))):
        audio_segment_path = os.path.join(audio_directory, file)
        
        # Dont run inference on the original audio file
        if audio_segment_path.split("/")[-1] != audio_file_name.split("/")[-1]:
            ds_process_audio(ds, audio_segment_path, file_handle)
            
    print("\nSRT file saved to", srt_file_name)
    file_handle.close()
        
if __name__ == "__main__":
    main()
