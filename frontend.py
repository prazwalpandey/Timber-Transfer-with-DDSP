import streamlit as st
import requests
import os
import numpy as np
from audio_recorder_streamlit import audio_recorder

def main():
    # App header
    st.markdown("<div style='text-align: center;'><h1>Music Stem Extraction</h1></div>", unsafe_allow_html=True)
    st.markdown("<h2>Upload Audio File</h2>", unsafe_allow_html=True)
    st.divider()  # Draws a horizontal line

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["mp3", "wav", "ogg", "flac"],
        key="upload_file",
        help="Supported formats: mp3, wav, ogg, flac.",
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

    # Instrument selection
    st.markdown("<h2>Select Instruments </h2>", unsafe_allow_html=True)
    st.divider()  # Draws a horizontal line

    # Checkboxes for instrument selection
    checkboxes = {
        "vocals": st.checkbox("Vocals 🎤", key="upload_vocals"),
        "drums": st.checkbox("Drums 🥁", key="upload_drums"),
        "bass": st.checkbox("Bass 🎸", key="upload_bass"),
        "other": st.checkbox("Other 🎶", key="upload_other"),
    }

    selected_instruments = [
        instrument for instrument, checked in checkboxes.items() if checked
    ]
    
    # Store selected instruments in session state
    if selected_instruments:
        st.session_state.selected_instruments = selected_instruments

    # Button to process the audio
    execute = st.button(
        "Process",
        type="primary",
        key="upload_separate_button",
        use_container_width=True,
    )

    if execute and uploaded_file is not None:
        # Process and call the API
        processing_message = st.empty()
        processing_message.info("Processing...")
        audio_file_path = download_uploaded_file(uploaded_file, "audio/upload")
        api_call_and_display(audio_file_path)
        processing_message.empty()

    # Show the results if they are available
    if 'results' in st.session_state:
        display_selected_instruments(st.session_state.selected_instruments)
    else:
        st.warning("No results yet. Upload an audio file and click 'Process'.")

    # Divider for the next section
    st.divider()

    # Call the tuning function
    tune()


def api_call_and_display(audio_file_path):
    """Calls the API and stores the results in session state."""
    api_url = "http://127.0.0.1:8080/separate_sota"
    absolute_path = os.path.abspath(audio_file_path)
    response = requests.post(url=api_url, json={"file_path": absolute_path})
    if response.status_code == 200:
        # Store results in session state to persist them across interactions
        st.session_state.results = response.json()
        st.success("Processing complete!")
    else:
        st.error("Failed to process the audio. Please try again.")


def display_selected_instruments(selected_instruments):
    """Display the results of the selected instruments."""
    if "results" in st.session_state:
        sample_rate = st.session_state.results["sr"]
        for stem in selected_instruments:
            # Get the path of each stem from the results
            stem_path = st.session_state.results.get(stem)
            if stem_path and os.path.isfile(stem_path):
                stem_array = np.load(stem_path).squeeze()  # Load the stem
                st.text(stem)
                st.audio(stem_array, sample_rate=sample_rate)
            else:
                st.error(f"Stem not found for {stem}.")


def download_uploaded_file(uploaded_file, output_folder):
    """Save the uploaded file locally."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_path = os.path.join(output_folder, "audio.mp3")  # Use a fixed name for simplicity
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def tune():
    """Secondary section for audio tuning."""
    st.title("Audio Processing with DDSP")

    # Dropdown for selecting input method
    input_method = st.selectbox(
        "Select Input Method:",
        ["Upload Audio", "Record Audio", "Select Stem"],
        help="Choose how you want to provide the audio.",
    )

    uploaded_tuning_file = None
    recorded_audio = None
    selected_stem = None

    if input_method == "Upload Audio":
        uploaded_tuning_file = st.file_uploader(
            "Upload a file for processing",
            type=["wav", "ogg", "mp3"],
            key="tuning_upload_file",
            help="Supported formats: wav, ogg, mp3.",
        )
        if uploaded_tuning_file:
            st.audio(uploaded_tuning_file, format="audio/wav")

    elif input_method == "Record Audio":
        st.write("Click below to start recording:")
        recorded_audio = st.audio_input("Record a voice message")
        if recorded_audio:
            st.audio(recorded_audio, format="audio/wav")
    elif input_method == "Select Stem":
        if "selected_instruments" in st.session_state:
            selected_instruments = st.session_state.selected_instruments
            selected_stem = st.selectbox(
                "Select the stem to process:",
                options=selected_instruments,
                help="Choose the stem from the uploaded file to process.",
            )

    # Audio tuning parameters
    st.markdown("<h2>Audio Tuning Parameters</h2>", unsafe_allow_html=True)
    # adjust=st.checkbox("Apply Filters", key="adjust")
    autotune = st.slider("Autotune (0-1)", 0.0, 1.0, 0.3, step=0.01, key="autotune")
    threshold = st.slider("Threshold (0.1-2.0)", 0.1, 2.0, 1.0, step=0.1, key="threshold")
    pitch_shift = st.slider("Pitch Shift (in octaves)", -2.0, 2.0, 1.0, step=0.1, key="pitch_shift")
    loudness_shift = st.slider("Loudness Shift (in dB)", -60, 0, -30, step=1, key="loudness_shift")
    reverb = st.slider("Reverb (0-1)", 0.0, 1.0, 0.2, step=0.1, key="reverb")

    execute_tune = st.button(
        "Process Tuning",
        type="primary",
        key="process_tune_button",
        use_container_width=True,
    )

    if execute_tune:
        file_to_process = None
        if uploaded_tuning_file:
            file_to_process = uploaded_tuning_file
        elif recorded_audio is not None:
            file_to_process = recorded_audio
        elif selected_stem:
            extracted_audio_path = f"./extracted/{selected_stem}.wav"
            if os.path.exists(extracted_audio_path):
                with open(extracted_audio_path, "rb") as f:
                    file_to_process = f.read()

        if file_to_process:
            with st.spinner("Processing audio..."):
                files = {"input_audio": file_to_process}
                response = requests.post(
                    "http://127.0.0.1:8000/process-audio/",
                    files=files,
                    data={
                        # "adjust": adjust,
                        "autotune": autotune,
                        "threshold": threshold,
                        "pitch_shift": pitch_shift,
                        "loudness_shift": loudness_shift,
                        "reverb": reverb,
                    },
                )

            if response.status_code == 200:
                output_file_path = "./output/output.wav"
                output_file_path1="./output/output1.wav"
                output_file_path2="./output/output2.wav"
                st.markdown("<h4>Audio Generated By Model</h4>", unsafe_allow_html=True)
                st.audio(output_file_path, format="audio/wav")
                st.markdown("<h4>Audio After Masking</h4>", unsafe_allow_html=True)
                st.audio(output_file_path1, format="audio/wav")
                st.markdown("<h4>Audio After Reverb</h4>", unsafe_allow_html=True)
                st.audio(output_file_path2, format="audio/wav")
            else:
                st.error("Audio processing failed.")
        else:
            st.error("No audio file available to process. Upload, record, or select a stem.")



    # Combine Processed Audio With Source Audio
    st.markdown("<h3>Combine Processed Audio With Source Audio</h3>", unsafe_allow_html=True)

    combine_options = {
        "model": st.radio("Output Option", options=["Output By Model", "Output After Masking", "Output After Reverb"], index=0, key="model"),
    }

    # Determine the selected output audio path based on the radio button choice
    if combine_options["model"] == "Output After Masking":
        selected_output_audio_path = "./output/output1.wav"
    elif combine_options["model"] == "Output After Reverb":
        selected_output_audio_path = "./output/output2.wav"
    else:
        selected_output_audio_path = "./output/output.wav"

    # Display the selected output audio
    if os.path.exists(selected_output_audio_path):
        st.markdown("**Selected Output Audio:**")
        st.audio(selected_output_audio_path, format="audio/wav")
    else:
        st.warning("No processed output audio found!")

    # Volume slider for adjusting output audio volume
    volume_level = st.slider("Adjust Output Audio Volume", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

    # Combine button
    combine = st.button(
        "Combine",
        type="primary",
        key="combine_button",
        use_container_width=True,
    )

    if combine:
        with st.spinner("Combining audio..."):
            combine_response = combine_audio_api_call(selected_output_audio_path, volume_level)
        
        if combine_response:
            combined_audio_path = "./combine/combined.wav"
            st.success("Audio combined successfully!")
            st.audio(combined_audio_path, format="audio/wav")
        else:
            st.error("Audio combining failed.")

# API Call function
def combine_audio_api_call(selected_output_audio_path, volume):
    """Calls the API to combine input and output audio with volume control."""
    combine_api_url = "http://127.0.0.1:8000/combine-audio/"
    input_audio_path = "./input/input.wav"
    
    # Ensure the combine directory exists
    combined_audio_path = "./combine/combined.wav"
    os.makedirs(os.path.dirname(combined_audio_path), exist_ok=True)

    files = {
        "input_audio": open(input_audio_path, "rb"),
        "output_audio": open(selected_output_audio_path, "rb"),  # Adjust this part to pass the file path based on the selected radio option
    }

    data = {"volume": volume}  # Send volume level to API

    try:
        response = requests.post(combine_api_url, files=files, data=data)
        if response.status_code == 200:
            with open(combined_audio_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error during audio combining: {e}")
        return False

        
if __name__ == "__main__":
    main()