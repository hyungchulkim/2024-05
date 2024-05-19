from tokenizers import Tokenizer

# Load the trained tokenizer
tokenizer1 = Tokenizer.from_file("korean_tokenizer.json")
tokenizer2 = Tokenizer.from_file("korean_subword_tokenizer.json")

# Function to tokenize a user's input prompt
def tokenize_input(prompt):
    # Tokenize the input prompt
    tokens = tokenizer1.encode(prompt)
    
    # Print the results
    print("Input Prompt:", prompt)
    print("Tokens:", tokens.tokens)
    print("Token IDs:", tokens.ids)

    # Tokenize the input prompt
    tokens = tokenizer2.encode(prompt)
    
    # Print the results
    print("Input Prompt:", prompt)
    print("Tokens:", tokens.tokens)
    print("Token IDs:", tokens.ids)

# User's input prompt
user_prompt = input("Enter a prompt in Korean: ")

# Tokenize the user's input prompt
tokenize_input(user_prompt)
