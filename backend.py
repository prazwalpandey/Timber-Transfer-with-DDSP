from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import os
import librosa
import numpy as np
import gin
import tensorflow as tf
from ddsp.training import models, metrics
from ddsp.spectral_ops import reset_crepe
import soundfile as sf
import glob
from pydub import AudioSegment
from audiomentations import Compose, AddBackgroundNoise, Gain
import ddsp
import pickle
from ddsp.training.postprocessing import (
    detect_notes, fit_quantile_transform
)
from scipy.signal import fftconvolve

app = FastAPI()

###########################Preprocess Audio And Helper Functions######################################
def restore_model(model_path, model):
    """Restore a trained model from the latest checkpoint."""
    latest_checkpoint = tf.train.latest_checkpoint(model_path)
    if not latest_checkpoint:
        raise FileNotFoundError(f"No valid checkpoint found at {model_path}")
    print(f"Restoring from {latest_checkpoint}")
    tf.train.Checkpoint(model=model).restore(
        latest_checkpoint).expect_partial()


def get_checkpoint_path(model_path):
    """Validate and normalize the model path."""
    model_path = os.path.normpath(model_path)
    if not glob.glob(os.path.join(model_path, "ckpt-*.index")):
        raise FileNotFoundError(f"No checkpoint files found in {model_path}")
    return model_path


def reshape_audio(audio):
    """Ensure audio has the correct shape for processing."""
    if len(audio.shape) == 1:
        return audio[np.newaxis, :]
    if len(audio.shape) > 2:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")
    return audio

# Helper Functions

def shift_ld(audio_features, ld_shift=-10.0):
  """Shift loudness by a number of ocatves."""
  audio_features['loudness_db'] += ld_shift
  return audio_features


def shift_f0(audio_features, pitch_shift=0.0):
  """Shift f0 by a number of ocatves."""
  audio_features['f0_hz'] *= 2.0 ** (pitch_shift)
  audio_features['f0_hz'] = np.clip(audio_features['f0_hz'], 
                                    0.0, 
                                    librosa.midi_to_hz(110.0))
  return audio_features

def get_tuning_factor(f0_midi, f0_confidence, mask_on):
  """Get an offset in cents, to most consistent set of chromatic intervals."""
  # Difference from midi offset by different tuning_factors.
  tuning_factors = np.linspace(-0.5, 0.5, 101)  # 1 cent divisions.
  midi_diffs = (f0_midi[mask_on][:, np.newaxis] -
                tuning_factors[np.newaxis, :]) % 1.0
  midi_diffs[midi_diffs > 0.5] -= 1.0
  weights = f0_confidence[mask_on][:, np.newaxis]

  ## Computes mininmum adjustment distance.
  cost_diffs = np.abs(midi_diffs)
  cost_diffs = np.mean(weights * cost_diffs, axis=0)

  ## Computes mininmum "note" transitions.
  f0_at = f0_midi[mask_on][:, np.newaxis] - midi_diffs
  f0_at_diffs = np.diff(f0_at, axis=0)
  deltas = (f0_at_diffs != 0.0).astype(float)
  cost_deltas = np.mean(weights[:-1] * deltas, axis=0)

  # Tuning factor is minimum cost.
  norm = lambda x: (x - np.mean(x)) / np.std(x)
  cost = norm(cost_deltas) + norm(cost_diffs)
  return tuning_factors[np.argmin(cost)]


def auto_tune(f0_midi, tuning_factor, mask_on, amount=0.0, chromatic=False):
  """Reduce variance of f0 from the chromatic or scale intervals."""
  if chromatic:
    midi_diff = (f0_midi - tuning_factor) % 1.0
    midi_diff[midi_diff > 0.5] -= 1.0
  else:
    major_scale = np.ravel(
        [np.array([0, 2, 4, 5, 7, 9, 11]) + 12 * i for i in range(10)])
    all_scales = np.stack([major_scale + i for i in range(12)])

    f0_on = f0_midi[mask_on]
    # [time, scale, note]
    f0_diff_tsn = (
        f0_on[:, np.newaxis, np.newaxis] - all_scales[np.newaxis, :, :])
    # [time, scale]
    f0_diff_ts = np.min(np.abs(f0_diff_tsn), axis=-1)
    # [scale]
    f0_diff_s = np.mean(f0_diff_ts, axis=0)
    scale_idx = np.argmin(f0_diff_s)
    scale = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb',
             'G', 'Ab', 'A', 'Bb', 'B', 'C'][scale_idx]

    # [time]
    f0_diff_tn = f0_midi[:, np.newaxis] - all_scales[scale_idx][np.newaxis, :]
    note_idx = np.argmin(np.abs(f0_diff_tn), axis=-1)
    midi_diff = np.take_along_axis(
        f0_diff_tn, note_idx[:, np.newaxis], axis=-1)[:, 0]
    print('Autotuning... \nInferred key: {}  '
          '\nTuning offset: {} cents'.format(scale, int(tuning_factor * 100)))

  # Adjust the midi signal.
  return f0_midi - amount * midi_diff

def apply_reverb(audio, sr, reverb_amount=0.3):
    """Applies artificial reverb using convolution with an impulse response."""
    ir_length = int(sr * 0.5)  # 0.5s impulse response
    ir = np.random.normal(0, 0.01, ir_length)
    reverb_audio = fftconvolve(audio, ir, mode='full')[: len(audio)]
    return (1 - reverb_amount) * audio + reverb_amount * reverb_audio

########################### Process Audio ######################################
@app.post("/process-audio/")
async def process_audio(
    input_audio: UploadFile = File(...),
    autotune: float = Form(0.3),
    threshold: float = Form(1.0),
    pitch_shift: float = Form(0.2),
    loudness_shift: int = Form(-30),
    reverb: float = Form(0.0),
):
    """Process input audio and generate output audio using the DDSP model."""
    model_path = "./models/sarangi"
    model_path = get_checkpoint_path(model_path)
    output_dir = "./output/"
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    output_audio_path = os.path.join(output_dir, "output.wav")
    output_audio_path1 = os.path.join(output_dir, "output1.wav")
    output_audio_path2 = os.path.join(output_dir, "output2.wav")  # Reverb output
    dataset_stats_file = os.path.join(model_path, "dataset_statistics.pkl")

    # Load dataset statistics
    try:
        if tf.io.gfile.exists(dataset_stats_file):
            with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
                DATASET_STATS = pickle.load(f)
    except Exception as err:
        print(f'Failed to load dataset statistics: {err}')
    print(DATASET_STATS)
    
    # Save uploaded audio file
    input_audio_path = "./input/input.wav"
    with open(input_audio_path, "wb") as f:
        audio_data = await input_audio.read()
        f.write(audio_data)

    # Load and preprocess audio
    audio, sr = librosa.load(input_audio_path, sr=16000)
    audio = reshape_audio(audio)
    reset_crepe()
    
    # Compute audio features
    audio_features = metrics.compute_audio_features(audio)
    audio_features["loudness_db"] = audio_features["loudness_db"].numpy().astype(np.float32)
    
    # Load model configuration
    gin_file = os.path.join(model_path, "operative_config-7000.gin")
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    # Calculate parameters for processing
    time_steps_train = gin.query_parameter("F0LoudnessPreprocessor.time_steps")
    n_samples_train = gin.query_parameter("Harmonic.n_samples")
    hop_size = n_samples_train // time_steps_train
    time_steps = audio.shape[1] // hop_size
    n_samples = time_steps * hop_size

    # Update configuration
    gin_params = [
        f"Harmonic.n_samples = {n_samples}",
        f"FilteredNoise.n_samples = {n_samples}",
        f"F0LoudnessPreprocessor.time_steps = {time_steps}",
        "oscillator_bank.use_angular_cumsum = True",
    ]
    with gin.unlock_config():
        gin.parse_config(gin_params)

    # Adjust audio features
    for key in ["f0_hz", "f0_confidence", "loudness_db"]:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features["audio"] = audio_features["audio"][:, :n_samples]

    # Load model and use the model
    model = models.Autoencoder()
    model.restore(model_path)


    #Output audio From the Model
    outputs = model(audio_features, training=False)
    audio_gen = model.get_audio_from_outputs(outputs).numpy()
    audio_gen = np.squeeze(audio_gen / np.max(np.abs(audio_gen)))  # Normalize
    sf.write(output_audio_path, audio_gen, sr, format="WAV", subtype="PCM_16")

    # Build model by running a batch through it.
    _ = model(audio_features, training=False)
    # Masking & Adjustments
    audio_features_mod = {k: v.copy() for k, v in audio_features.items()}
    print(audio_features_mod)

    mask_on, note_on_value = None, None
    quiet=40

    if True:
        mask_on, note_on_value = detect_notes(audio_features['loudness_db'],
                                            audio_features['f0_confidence'],
                                            threshold)
    if np.any(mask_on):

        _ , loudness_norm = fit_quantile_transform(
            audio_features['loudness_db'],
            mask_on,
            inv_quantile=DATASET_STATS['quantile_transform'])
        
        mask_off = np.logical_not(mask_on)
        loudness_norm[mask_off] -= quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
        loudness_norm=np.reshape(loudness_norm, audio_features['loudness_db'].shape)

        audio_features_mod['loudness_db'] = loudness_norm

        if autotune:
            # Apply Auto-tune
            f0_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod['f0_hz']))
            tuning_factor = get_tuning_factor(f0_midi, audio_features_mod['f0_confidence'], mask_on)
            f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=autotune)
            audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)
    else:
        print('\nSkipping auto-adujst (box not checked or no dataset statistics found).')

    # Apply manual pitch and loudness shift
    audio_features_mod = shift_ld(audio_features_mod, loudness_shift)
    audio_features_mod = shift_f0(audio_features_mod, pitch_shift)
    print(audio_features_mod)
        
    af=audio_features if audio_features_mod is None else audio_features_mod

    outputs = model(af, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)
    audio_gen = audio_gen.numpy()
    audio_gen=np.squeeze(audio_gen / np.max(np.abs(audio_gen)))  # Normalize
    sf.write(output_audio_path1, audio_gen, sr, format="WAV", subtype="PCM_16")
    print(f"Masked audio saved to {output_audio_path1}")
    
    # Apply Reverb
    audio_reverb = apply_reverb(audio_gen, sr, reverb_amount=reverb)
    sf.write(output_audio_path2, audio_reverb, sr, format="WAV", subtype="PCM_16")
    
    return FileResponse(output_audio_path, media_type="audio/wav", filename="output.wav")

    
###########################Combine Audio######################################

@app.post("/combine-audio/")
async def combine_audio(
    input_audio: UploadFile = File(...), 
    output_audio: UploadFile = File(...),
    volume: float = Form(1.0)  # Default volume is 1.0 (no change)
):
    """Combine two audio files by mixing them, applying volume adjustment."""
    try:
        # Paths for temporary storage
        input_path = "./temp/input_audio.wav"
        output_path = "./temp/output_audio.wav"
        combined_path = "./combine/combined.wav"

        os.makedirs("./temp", exist_ok=True)
        os.makedirs("./combine", exist_ok=True)

        # Save uploaded files
        with open(input_path, "wb") as f:
            f.write(await input_audio.read())
        with open(output_path, "wb") as f:
            f.write(await output_audio.read())

        # Load audio files
        input_sound = AudioSegment.from_file(input_path)
        output_sound = AudioSegment.from_file(output_path)

        # Ensure both have the same frame rate and channels
        if input_sound.frame_rate != output_sound.frame_rate:
            input_sound = input_sound.set_frame_rate(output_sound.frame_rate)
        if input_sound.channels != output_sound.channels:
            input_sound = input_sound.set_channels(output_sound.channels)

        # Adjust output audio volume
        output_sound = output_sound + (20 * (volume - 1))  # Convert to dB scale

        # Pad shorter audio to match the longer one
        if len(input_sound) > len(output_sound):
            output_sound += AudioSegment.silent(duration=len(input_sound) - len(output_sound))
        elif len(output_sound) > len(input_sound):
            input_sound += AudioSegment.silent(duration=len(output_sound) - len(input_sound))

        # Mix both audios
        combined_sound = input_sound.overlay(output_sound)

        # Export the combined file
        combined_sound.export(combined_path, format="wav")

        # Clean up temporary files
        os.remove(input_path)
        os.remove(output_path)

        return FileResponse(combined_path, media_type="audio/wav", filename="combined.wav")

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}