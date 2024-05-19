from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# Initialize a tokenizer with a Byte-Pair Encoding (BPE) model
tokenizer = Tokenizer(BPE())

# Set up a trainer for the BPE model
trainer = BpeTrainer(vocab_size=32000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>"
])

# Pre-tokenizer to split input text into words
tokenizer.pre_tokenizer = Whitespace()

# List of Korean text files for training the tokenizer
files = ["./korean_text_only.json"]

# Train the tokenizer on the provided files
tokenizer.train(files, trainer)

# Post-processing template for the tokenizer
tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[
        ("<s>", 0),
        ("</s>", 2),
    ],
)

# Save the tokenizer to disk
tokenizer.save("korean_tokenizer.json")

print("Tokenizer training completed and saved as 'korean_tokenizer.json'.")
