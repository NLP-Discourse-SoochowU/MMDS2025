from collections import Counter
import numpy as np
import zlib

def compression_ratio(text, get_ratio=False, get_ori=False, get_com=False):
    """
    Calculate the compression ratio of a given text using zlib compression.
    Args:
        text (str): The input text to compress.
    
    Returns:
        float: The compression ratio (compressed size / original size).
        int: The size of the original text (in bytes).
        int: The size of the compressed text (in bytes).
    """
    if not isinstance(text, str) or not text:
        raise ValueError("Input must be a non-empty string.")
    
    # Convert text to bytes
    original_bytes = text.encode('utf-8')
    original_size = len(original_bytes)
    
    # Compress the text
    compressed_bytes = zlib.compress(original_bytes)
    compressed_size = len(compressed_bytes)
    
    # Calculate the compression ratio
    ratio = compressed_size / original_size
    
    results = tuple()
    if get_ratio:
        results = results + (ratio,)
    if get_ori:
        results = results + (original_size,)
    if get_com:
        results = results + (compressed_size,)
    return results


def entropy(text):
    # Step 1: Tokenize text into characters
    chars = list(text)
    # Step 2: Calculate the frequency of each character
    char_counts = Counter(chars)
    total_chars = len(chars)
    # Step 3: Compute the probability of each character
    probabilities = np.array([count / total_chars for count in char_counts.values()])
    # Step 4: Compute the entropy using the formula
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

# Example Usage
if __name__ == "__main__":
    sample_text = "This is an example of a text to measure the compression ratio. The compression ratio reflects the information density of the text."
    ratio, original_size, compressed_size = compression_ratio(sample_text)
    print(f"Original Size: {original_size} bytes")
    print(f"Compressed Size: {compressed_size} bytes")
    print(f"Compression Ratio: {ratio:.3f}")
