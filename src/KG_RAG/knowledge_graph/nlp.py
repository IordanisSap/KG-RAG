import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def lemmatize_word(word: str) -> str:
    """
    Lemmatizes a given word using spaCy.

    Args:
        word (str): The word to lemmatize.

    Returns:
        str: The lemmatized form of the word.
    """
    doc = nlp(word)
    # Return the lemma of the first (and only) token
    return doc[0].lemma_


if __name__ == "__main__":
    # Example usage
    print(lemmatize_word("directed"))  # Output: "direct"
    print(lemmatize_word("running"))   # Output: "run"
    print(lemmatize_word("lives"))     # Output: "live"
