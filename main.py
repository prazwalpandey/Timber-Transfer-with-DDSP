###################################### Imports #####################################################

import sys
import numpy as np
import torchaudio
from pydantic import BaseModel
from pathlib import Path
from fastapi import FastAPI, HTTPException
from openunmix import predict
import soundfile as sf

################## Setting Up the Application and Save Directory ###################################

sys.path.append("../")
app = FastAPI()
save_path = Path("save/")
extracted_path = Path("extracted/")  # Path for extracted .wav files

save_path.mkdir(parents=True, exist_ok=True)
extracted_path.mkdir(parents=True, exist_ok=True)

################################# Data Model Definition ############################################
class AudioRequest(BaseModel):
    file_path: str


################################## separate_ummix Function #########################################
################################## separate_ummix Function #########################################
def separate_ummix(waveform, sample_rate):
    # Use Open-Unmix to separate the waveform into stems
    estimates = predict.separate(
        audio=waveform,
        rate=sample_rate,
    )
    return estimates


############################### API Endpoint Definition ############################################
@app.post("/separate_sota")
async def separate_sota_by_path(audio_request: AudioRequest):
    # Validate if the provided file path exists
    audio_file_path = Path(audio_request.file_path)
    if not audio_file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Load the audio file into waveform and sample rate
    waveform, sample_rate = torchaudio.load(audio_file_path)

    # Separate audio into stems using Open-Unmix
    separated_audio = separate_ummix(waveform, sample_rate)

    # Prepare result dictionary with sample rate
    result = {"sr": sample_rate}

    # Process each stem
    for stem in ["vocals", "drums", "bass", "other"]:
        stem_waveform = separated_audio[stem]

        # Save the stem as a .npy file in the save_path directory
        npy_file_path = save_path / f"{stem}.npy"
        np.save(npy_file_path, stem_waveform)

        # Convert the waveform to NumPy array and squeeze extra dimensions
        stem_waveform_np = np.squeeze(stem_waveform.numpy().astype(np.float32))

        # Save the stem as a .wav file in the extracted directory
        wav_file_path = extracted_path / f"{stem}.wav"
        sf.write(
            wav_file_path,
            stem_waveform_np.T,  # Ensure (samples, channels) format
            sample_rate,
            format="WAV"
        )

        # Add the paths to the result dictionary
        result[stem] = str(npy_file_path.resolve())

    return result