from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# Initialize a tokenizer with a WordPiece model
tokenizer = Tokenizer(WordPiece())

# Set up a trainer for the WordPiece model
trainer = WordPieceTrainer(vocab_size=32000, min_frequency=2, special_tokens=[
    "[CLS]",
    "[PAD]",
    "[SEP]",
    "[UNK]",
    "[MASK]"
])

# Pre-tokenizer to split input text into words
tokenizer.pre_tokenizer = Whitespace()

# List of Korean text files for training the tokenizer
files = ["./korean_text_only.json"]

# Train the tokenizer on the provided files
tokenizer.train(files, trainer)

# Post-processing template for the tokenizer
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[
        ("[CLS]", 0),
        ("[SEP]", 2),
    ],
)

# Save the tokenizer to disk
tokenizer.save("korean_subword_tokenizer.json")

print("Tokenizer training completed and saved as 'korean_subword_tokenizer.json'.")
