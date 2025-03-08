@app.post("/separate_karaoke")
async def separate_karaoke_by_path(audio_request: AudioRequest):
    audio_file_path = audio_request.file_path
    if not os.path.isfile(audio_file_path):
        raise HTTPException(status_code=404, detail="File not found")
    waveform, sample_rate = torchaudio.load(audio_file_path)
    separated_audio = extract_vocal(waveform, sample_rate)
    instrument = separated_audio["residual"]
    file_path = save_path / "residual.npy"
    np.save(file_path, instrument)
    result = {"sr": sample_rate, "residual": f"api/{file_path}"}
    return result


def extract_vocal(uploaded_audio_path):
    api_url = "http://127.0.0.1:8000/separate_karaoke"
    absolute_test_path = os.path.abspath(uploaded_audio_path)
    response = requests.post(url=api_url, json={"file_path": absolute_test_path})
    if response.status_code == 200:
        st.info(f"Separated the vocal from audio")
        display_vocal_extract(response)
        # st.success("Processing complete!")
    else:
        st.error("Failed to process the audio. Please try again.")