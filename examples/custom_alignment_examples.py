"""
Test script for custom alignment models feature

This script demonstrates how to use custom alignment models with WhisperX Worker.
"""

# Example 1: Using a custom German alignment model
example_german = {
    "input": {
        "audio_file": "https://example.com/german_audio.wav",
        "language": "de",
        "align_output": True,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
        "batch_size": 32
    }
}

# Example 2: Using a custom Japanese alignment model
example_japanese = {
    "input": {
        "audio_file": "https://example.com/japanese_audio.wav",
        "language": "ja",
        "align_output": True,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        "batch_size": 32
    }
}

# Example 3: Custom model with diarization
example_with_diarization = {
    "input": {
        "audio_file": "https://example.com/french_meeting.wav",
        "language": "fr",
        "align_output": True,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        "diarization": True,
        "huggingface_access_token": "YOUR_HF_TOKEN",
        "min_speakers": 2,
        "max_speakers": 4,
        "batch_size": 32
    }
}

# Example 4: Custom model for unsupported language (Urdu)
example_urdu = {
    "input": {
        "audio_file": "https://example.com/urdu_audio.wav",
        "language": "ur",
        "align_output": True,
        "custom_align_model": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
        "batch_size": 16
    }
}

# Example 5: Using torchaudio model
example_torchaudio = {
    "input": {
        "audio_file": "https://example.com/english_audio.wav",
        "language": "en",
        "align_output": True,
        "custom_align_model": "WAV2VEC2_ASR_BASE_960H",
        "batch_size": 64
    }
}

# Example 6: Full configuration with speaker verification and custom alignment
example_full = {
    "input": {
        "audio_file": "https://example.com/interview.wav",
        "language": "es",
        "align_output": True,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
        "diarization": True,
        "speaker_verification": True,
        "speaker_samples": [
            {
                "name": "Host",
                "url": "https://example.com/host_sample.wav"
            },
            {
                "name": "Guest",
                "url": "https://example.com/guest_sample.wav"
            }
        ],
        "huggingface_access_token": "YOUR_HF_TOKEN",
        "min_speakers": 2,
        "max_speakers": 2,
        "batch_size": 32,
        "temperature": 0.2,
        "debug": True
    }
}

if __name__ == "__main__":
    import json
    
    print("Custom Alignment Model Examples")
    print("=" * 50)
    
    examples = [
        ("German", example_german),
        ("Japanese", example_japanese),
        ("With Diarization", example_with_diarization),
        ("Urdu (Unsupported Language)", example_urdu),
        ("Torchaudio Model", example_torchaudio),
        ("Full Configuration", example_full)
    ]
    
    for name, example in examples:
        print(f"\n{name}:")
        print(json.dumps(example, indent=2))
    
    print("\n" + "=" * 50)
    print("\nTo use these examples with RunPod:")
    print("1. Deploy the WhisperX Worker on RunPod")
    print("2. Send a POST request with one of the above JSON payloads")
    print("3. Check the output for word-level timestamps and speaker information")
    print("\nFor more information, see CUSTOM_ALIGNMENT_MODELS.md")
