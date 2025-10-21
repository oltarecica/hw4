def count_word(text, word):
    """Count how many times a word appears in a text."""
    return text.lower().split().count(word.lower())
