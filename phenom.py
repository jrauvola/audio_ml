import nltk
from nltk.corpus import cmudict
import re

nltk.download('cmudict')

def get_phonemes(word):
    pron_dict = cmudict.dict()
    
    word = word.lower()
  
    phonemes = pron_dict.get(word)
    
    # Return the phonemes if the word is found, otherwise return a message
    if phonemes:
        return phonemes[0]  # return the first pronunciation variant
    else:
        return f"No phoneme representation found for '{word}'."


sentence = "The quick brown fox jumps over the lazy dog."
print("Sentence:", sentence)
words = sentence.split()

#remove non-alphabetic characters using regex
word_regex = re.compile(r'[^a-zA-Z]')
words = [word_regex.sub('', word) for word in words]

print("Words:", words)
sentence_phonemes = [get_phonemes(word) for word in words]
print("Phonemes:", sentence_phonemes)

