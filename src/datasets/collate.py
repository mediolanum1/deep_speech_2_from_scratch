import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    # Unzip by extracting values based on dictionary keys
    audios, spectrograms, texts, text_encodeds, audio_paths = zip(
        *[(item["audio"], item["spectrogram"], item["text"], item["text_encoded"], item["audio_path"]) for item in dataset_items]
    )

    lengths = []
    for spectrogram in spectrograms: 
        lengths.append(spectrogram.shape[-1])

    spectrograms_padded = torch.zeros(len(lengths), 128, max(lengths))

    for i, spectrogram in enumerate(spectrograms):
        spectrograms_padded[i, :, :lengths[i]] = spectrogram

    text_lengths = []
    
    for text in text_encodeds: 
        text_lengths.append(text.shape[-1])
    
    text_padded = torch.zeros(len(text_lengths), max(text_lengths))
    for i, text in enumerate(text_encodeds):
        text_padded[i, :text_lengths[i]] = text

    return {
        'audio': audios,
        'spectrogram': spectrograms_padded,
        'spectrogram_length': lengths,
        'text': texts,
        'text_encoded': text_padded,
        'text_encoded_length': torch.tensor(text_lengths),
        'audio_path': audio_paths,
    }