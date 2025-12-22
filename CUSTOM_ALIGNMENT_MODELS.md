# Custom Alignment Models Guide

## Overview

WhisperX Worker now supports custom alignment models, allowing you to use specialized wav2vec2 models for improved word-level timestamp accuracy. This is especially useful for:

-   Languages not in the default supported list
-   Specific dialects or accents
-   Domain-specific audio (medical, legal, technical, etc.)
-   Improved accuracy for supported languages

## How It Works

By default, WhisperX uses pre-configured alignment models for common languages (English, French, German, Spanish, Italian, etc.). However, you can now override this by specifying your own alignment model from Hugging Face or torchaudio.

The alignment process:

1. **Transcription**: Whisper performs speech-to-text transcription
2. **Alignment**: A wav2vec2 model aligns the transcription to get precise word-level timestamps
3. **Output**: Returns segments with accurate start/end times for each word

## Usage

### Basic Example

```json
{
    "input": {
        "audio_file": "https://example.com/audio.wav",
        "align_output": true,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    }
}
```

### With Full Configuration

```json
{
    "input": {
        "audio_file": "https://example.com/audio.wav",
        "language": "de",
        "align_output": true,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
        "batch_size": 32,
        "temperature": 0.2,
        "debug": true
    }
}
```

## Finding Alignment Models

### From Hugging Face

1. Go to [Hugging Face Models](https://huggingface.co/models)
2. Filter by:
    - Task: `Automatic Speech Recognition`
    - Library: `transformers`
    - Search for: `wav2vec2` + your language
3. Look for models with high downloads and good documentation

**Recommended model patterns:**

-   `jonatasgrosman/wav2vec2-large-xlsr-53-{language}` - Excellent multilingual models
-   `facebook/wav2vec2-large-{language}` - Official Facebook models
-   Language-specific community models

### From Torchaudio

You can also use torchaudio pipeline models:

-   `WAV2VEC2_ASR_BASE_960H` (English)
-   `VOXPOPULI_ASR_BASE_10K_FR` (French)
-   `VOXPOPULI_ASR_BASE_10K_DE` (German)
-   `VOXPOPULI_ASR_BASE_10K_ES` (Spanish)
-   `VOXPOPULI_ASR_BASE_10K_IT` (Italian)

## Popular Models by Language

### European Languages

| Language            | Model                                              | Description                 |
| ------------------- | -------------------------------------------------- | --------------------------- |
| German              | `jonatasgrosman/wav2vec2-large-xlsr-53-german`     | High-quality German ASR     |
| French              | `jonatasgrosman/wav2vec2-large-xlsr-53-french`     | High-quality French ASR     |
| Spanish             | `jonatasgrosman/wav2vec2-large-xlsr-53-spanish`    | High-quality Spanish ASR    |
| Italian             | `jonatasgrosman/wav2vec2-large-xlsr-53-italian`    | High-quality Italian ASR    |
| Portuguese          | `jonatasgrosman/wav2vec2-large-xlsr-53-portuguese` | Portuguese (both PT and BR) |
| Dutch               | `jonatasgrosman/wav2vec2-large-xlsr-53-dutch`      | High-quality Dutch ASR      |
| Polish              | `jonatasgrosman/wav2vec2-large-xlsr-53-polish`     | High-quality Polish ASR     |
| Russian             | `jonatasgrosman/wav2vec2-large-xlsr-53-russian`    | High-quality Russian ASR    |
| Greek               | `jonatasgrosman/wav2vec2-large-xlsr-53-greek`      | High-quality Greek ASR      |
| Finnish             | `jonatasgrosman/wav2vec2-large-xlsr-53-finnish`    | High-quality Finnish ASR    |
| Hungarian           | `jonatasgrosman/wav2vec2-large-xlsr-53-hungarian`  | High-quality Hungarian ASR  |
| Czech               | `comodoro/wav2vec2-xls-r-300m-cs-250`              | Czech language model        |
| Swedish             | `KBLab/wav2vec2-large-voxrex-swedish`              | Swedish language model      |
| Norwegian (Bokmål)  | `NbAiLab/nb-wav2vec2-1b-bokmaal-v2`                | Norwegian Bokmål            |
| Norwegian (Nynorsk) | `NbAiLab/nb-wav2vec2-1b-nynorsk`                   | Norwegian Nynorsk           |
| Turkish             | `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish`          | Turkish language model      |
| Ukrainian           | `Yehor/wav2vec2-xls-r-300m-uk-with-small-lm`       | Ukrainian language model    |

### Asian Languages

| Language   | Model                                                 | Description                   |
| ---------- | ----------------------------------------------------- | ----------------------------- |
| Japanese   | `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`      | High-quality Japanese ASR     |
| Chinese    | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | Mandarin Chinese (Simplified) |
| Korean     | `kresnik/wav2vec2-large-xlsr-korean`                  | Korean language model         |
| Vietnamese | `nguyenvulebinh/wav2vec2-base-vi-vlsp2020`            | Vietnamese language model     |
| Hindi      | `theainerd/Wav2Vec2-large-xlsr-hindi`                 | Hindi language model          |
| Telugu     | `anuragshas/wav2vec2-large-xlsr-53-telugu`            | Telugu language model         |
| Malayalam  | `gvs/wav2vec2-large-xlsr-malayalam`                   | Malayalam language model      |

### Middle Eastern Languages

| Language | Model                                           | Description                    |
| -------- | ----------------------------------------------- | ------------------------------ |
| Arabic   | `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`  | High-quality Arabic ASR        |
| Persian  | `jonatasgrosman/wav2vec2-large-xlsr-53-persian` | Persian (Farsi) language model |
| Hebrew   | `imvladikon/wav2vec2-xls-r-300m-hebrew`         | Hebrew language model          |
| Urdu     | `kingabzpro/wav2vec2-large-xls-r-300m-Urdu`     | Urdu language model            |

### Other Languages

| Language | Model                                             | Description             |
| -------- | ------------------------------------------------- | ----------------------- |
| Catalan  | `softcatala/wav2vec2-large-xlsr-catala`           | Catalan language model  |
| Basque   | `stefan-it/wav2vec2-large-xlsr-53-basque`         | Basque language model   |
| Galician | `ifrz/wav2vec2-large-xlsr-galician`               | Galician language model |
| Filipino | `Khalsuu/filipino-wav2vec2-l-xls-r-300m-official` | Filipino/Tagalog model  |

## Benefits of Custom Models

### 1. Language Coverage

Enable alignment for languages not in the default list:

```json
{
    "input": {
        "audio_file": "https://example.com/urdu_audio.wav",
        "language": "ur",
        "align_output": true,
        "custom_align_model": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu"
    }
}
```

### 2. Improved Accuracy

Use specialized models for better results:

```json
{
    "input": {
        "audio_file": "https://example.com/technical_german.wav",
        "language": "de",
        "align_output": true,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    }
}
```

### 3. Dialect Support

Some models are fine-tuned on specific dialects:

```json
{
    "input": {
        "audio_file": "https://example.com/brazilian_portuguese.wav",
        "language": "pt",
        "align_output": true,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"
    }
}
```

## Technical Details

### Model Requirements

The custom alignment model must be:

-   A wav2vec2-based model for CTC (Connectionist Temporal Classification)
-   Available on Hugging Face or torchaudio
-   Compatible with the `transformers` library or PyTorch torchaudio

### How to Verify a Model

Before using a custom model, check:

1. **Model Card**: Read the model description on Hugging Face
2. **Language**: Ensure it matches your audio language
3. **Sample Rate**: Most models expect 16kHz audio (WhisperX handles this automatically)
4. **Performance Metrics**: Check WER (Word Error Rate) if available

### Example Model Card Check

Visit the model page (e.g., https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german) and verify:

-   ✅ Task: Automatic Speech Recognition
-   ✅ Library: transformers
-   ✅ Language: German (de)
-   ✅ Base model: wav2vec2-large-xlsr-53

## Troubleshooting

### Model Not Found

```
Error: The chosen align_model "xyz" could not be found...
```

**Solution**: Verify the model name is correct and exists on Hugging Face.

### Language Mismatch

If the detected language doesn't match your custom model's language, alignment may fail or produce poor results.

**Solution**: Explicitly specify the language:

```json
{
    "language": "de",
    "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-german"
}
```

### Memory Issues

Large alignment models may require more GPU memory.

**Solution**: Reduce batch size or use a smaller model variant:

```json
{
    "batch_size": 16,
    "custom_align_model": "facebook/wav2vec2-base-960h"
}
```

## Performance Considerations

### Model Size vs. Accuracy

-   **Large models** (e.g., `large-xlsr-53`): Better accuracy, more GPU memory, slower
-   **Base models**: Faster, less memory, slightly lower accuracy

### Batch Size

When using custom alignment models, you may need to adjust batch size based on:

-   Model size
-   GPU memory available
-   Audio duration

### Caching

Models are downloaded and cached on first use. Subsequent requests using the same model will be faster.

## Advanced Usage

### Combining with Diarization

```json
{
    "input": {
        "audio_file": "https://example.com/meeting.wav",
        "language": "de",
        "align_output": true,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
        "diarization": true,
        "huggingface_access_token": "YOUR_TOKEN",
        "min_speakers": 2,
        "max_speakers": 5
    }
}
```

### With Speaker Verification

```json
{
    "input": {
        "audio_file": "https://example.com/interview.wav",
        "language": "fr",
        "align_output": true,
        "custom_align_model": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        "diarization": true,
        "speaker_verification": true,
        "speaker_samples": [
            {
                "name": "Interviewer",
                "url": "https://example.com/interviewer_sample.wav"
            },
            {
                "name": "Guest",
                "url": "https://example.com/guest_sample.wav"
            }
        ],
        "huggingface_access_token": "YOUR_TOKEN"
    }
}
```

## Resources

-   [WhisperX GitHub](https://github.com/m-bain/whisperX)
-   [Hugging Face ASR Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition)
-   [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
-   [WhisperX Paper](https://arxiv.org/abs/2303.00747)

## Contributing

Found a great alignment model for your language? Consider:

1. Testing it with this worker
2. Documenting your results
3. Sharing with the community via a PR to update this guide
