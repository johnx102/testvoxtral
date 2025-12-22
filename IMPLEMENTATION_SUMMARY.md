# Custom Alignment Models Feature - Implementation Summary

## Overview

Added support for custom alignment models to WhisperX Worker, allowing users to specify their own wav2vec2 models from Hugging Face or torchaudio for improved word-level timestamp accuracy.

## Changes Made

### 1. Core Implementation Files

#### `src/rp_schema.py`

-   **Added parameter**: `custom_align_model` (string, optional, default: None)
-   This parameter is validated through the existing RunPod validation system

#### `src/predict.py`

-   **Modified `predict()` method**:
    -   Added `custom_align_model` parameter with detailed description
    -   Updated alignment logic to accept custom models even for unsupported languages
-   **Modified `align()` function**:

    -   Added `custom_align_model` parameter (default: None)
    -   Passes custom model name to `whisperx.load_align_model()`
    -   Updated function signature: `align(audio, result, debug, custom_align_model=None)`

-   **Enhanced error messaging**:
    -   Improved error message to suggest using custom alignment models when language isn't supported

### 2. Documentation Files

#### `README.md`

-   Added `custom_align_model` parameter to the Input Parameters table
-   Added "Custom alignment models support" to Features list
-   Added usage example section "Custom Alignment Model" with:
    -   Basic example showing how to use a custom model
    -   List of popular models by language
    -   Link to comprehensive guide
    -   Note about enabling alignment for unsupported languages

#### `CUSTOM_ALIGNMENT_MODELS.md` (NEW)

Comprehensive 300+ line guide covering:

-   Overview and benefits of custom alignment models
-   How it works (transcription → alignment → output)
-   Usage examples for various scenarios
-   Finding and selecting models from Hugging Face
-   Popular models organized by language (40+ languages)
-   Technical details and requirements
-   Troubleshooting common issues
-   Performance considerations
-   Advanced usage examples
-   Resources and links

#### `examples/custom_alignment_examples.py` (NEW)

Python script with 6 complete examples:

1. German custom model
2. Japanese custom model
3. Custom model with diarization
4. Unsupported language (Urdu)
5. Torchaudio model
6. Full configuration with speaker verification

### 3. Directory Structure

```
whisperx-worker/
├── src/
│   ├── predict.py          [MODIFIED]
│   └── rp_schema.py        [MODIFIED]
├── examples/               [NEW]
│   └── custom_alignment_examples.py
├── CUSTOM_ALIGNMENT_MODELS.md [NEW]
└── README.md              [MODIFIED]
```

## Technical Details

### How It Works

1. **User Input**: User specifies `custom_align_model` parameter with a model name (e.g., "jonatasgrosman/wav2vec2-large-xlsr-53-german")

2. **Model Loading**: The system passes this to `whisperx.load_align_model()`:

    ```python
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
        model_name=custom_align_model
    )
    ```

3. **Fallback Behavior**:

    - If `custom_align_model` is None, WhisperX uses default models
    - If specified, overrides the default model selection

4. **Language Support Expansion**:
    - Before: Only languages in DEFAULT_ALIGN_MODELS_TORCH or DEFAULT_ALIGN_MODELS_HF
    - Now: Any language if a custom model is provided

### API Compatibility

**Backward Compatible**: Yes ✓

-   Existing requests without `custom_align_model` work exactly as before
-   New parameter is optional with default value None

**Forward Compatible**: Yes ✓

-   Easy to extend with additional model-related parameters in the future

### Supported Model Sources

1. **Hugging Face Models**:

    - Format: `username/model-name`
    - Example: `jonatasgrosman/wav2vec2-large-xlsr-53-german`
    - Uses `transformers` library

2. **Torchaudio Models**:
    - Format: Model constant name
    - Example: `WAV2VEC2_ASR_BASE_960H`
    - Uses PyTorch torchaudio pipelines

## Use Cases Enabled

### 1. Unsupported Languages

Enable alignment for languages not in default list:

-   Urdu, Telugu, Malayalam, Vietnamese, etc.

### 2. Improved Accuracy

Use specialized models for better results:

-   Domain-specific models (medical, legal, technical)
-   Accent-specific models
-   Dialect-specific models

### 3. Research & Experimentation

Test different alignment models for optimal results

### 4. Future-Proofing

Use newer/better models as they become available without code changes

## Example API Requests

### Basic Usage

```json
{
    "input": {
        "audio_file": "https://example.com/audio.wav",
        "align_output": true,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    }
}
```

### With All Features

```json
{
    "input": {
        "audio_file": "https://example.com/meeting.wav",
        "language": "fr",
        "align_output": true,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        "diarization": true,
        "speaker_verification": true,
        "speaker_samples": [{ "name": "Speaker1", "url": "https://example.com/speaker1.wav" }],
        "huggingface_access_token": "hf_...",
        "batch_size": 32
    }
}
```

## Testing Recommendations

### Unit Tests

-   Test with valid Hugging Face model
-   Test with valid torchaudio model
-   Test with invalid model name (should fail gracefully)
-   Test with None (default behavior)
-   Test with unsupported language + custom model

### Integration Tests

-   End-to-end transcription with custom German model
-   End-to-end transcription with custom Japanese model
-   Verify word-level timestamps are accurate
-   Test with diarization enabled
-   Test with speaker verification enabled

### Performance Tests

-   Compare default vs. custom model performance
-   Memory usage with different model sizes
-   Batch size optimization with custom models

## Known Limitations

1. **Model Must Be Compatible**:

    - Must be wav2vec2-based CTC model
    - Must be on Hugging Face or torchaudio

2. **GPU Memory**:

    - Large models require more memory
    - May need to reduce batch size

3. **First Request Slower**:

    - Model download happens on first use
    - Subsequent requests are cached

4. **Language Mismatch**:
    - Best results when model language matches audio language
    - User should specify language explicitly for best results

## Future Enhancements

Potential improvements for future versions:

1. **Model Validation**:

    - Pre-validate model exists before attempting to load
    - Better error messages for common issues

2. **Model Caching**:

    - Pre-download popular models during container build
    - Reduce cold start time

3. **Model Recommendations**:

    - Suggest best model for detected language
    - Auto-select best model based on language and audio characteristics

4. **Performance Metrics**:

    - Return alignment confidence scores
    - Log model performance metrics

5. **Multi-Model Support**:
    - Allow different models for different segments
    - Ensemble alignment for improved accuracy

## Benefits to Users

1. **Flexibility**: Use any compatible model from Hugging Face
2. **Accuracy**: Choose models optimized for specific use cases
3. **Language Coverage**: Support for 40+ languages beyond defaults
4. **Future-Proof**: Easy to adopt new models as they're released
5. **No Code Changes**: Simple parameter addition to existing workflow

## Migration Guide

For existing users:

1. No changes required - feature is opt-in
2. Add `custom_align_model` parameter to enable
3. Refer to CUSTOM_ALIGNMENT_MODELS.md for model selection

## Resources

-   [WhisperX Documentation](https://github.com/m-bain/whisperX)
-   [Hugging Face ASR Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition)
-   [WhisperX Paper](https://arxiv.org/abs/2303.00747)
-   [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)

## Credits

-   Based on WhisperX by Max Bain et al.
-   Uses models from Hugging Face community (especially jonatasgrosman's excellent multilingual models)
-   Powered by Facebook's wav2vec2 technology
