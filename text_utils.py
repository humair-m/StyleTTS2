# IPA Phonemizer: https://github.com/bootphon/phonemizer

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Create symbol to index mapping
dicts = {}
for i in range(len(symbols)):
    dicts[symbols[i]] = i

class TextCleaner:
    """
    Text cleaner for converting text to token indices.
    """
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        print(f"Vocabulary size: {len(dicts)}")
        
    def __call__(self, text):
        """
        Convert text to list of token indices.
        
        Args:
            text (str): Input text to convert
            
        Returns:
            list: List of token indices
        """
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(f"Unknown character in text: {text}")
                # Optionally, you could add a default token for unknown characters
        return indexes
    
    def get_vocab_size(self):
        """Get the vocabulary size."""
        return len(self.word_index_dictionary)
    
    def decode(self, indices):
        """
        Convert list of indices back to text.
        
        Args:
            indices (list): List of token indices
            
        Returns:
            str: Decoded text
        """
        # Create reverse mapping
        reverse_dict = {v: k for k, v in self.word_index_dictionary.items()}
        return ''.join([reverse_dict.get(idx, '') for idx in indices])
