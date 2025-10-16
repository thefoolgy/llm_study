def save_vocab_simple(vocab, filename="vocab.txt"):
    """Save vocabulary to a simple text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for token_id, token_bytes in vocab.items():
            # Convert bytes to string representation
            if isinstance(token_bytes, bytes):
                # Try to decode, fallback to repr if it fails
                try:
                    token_str = token_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    token_str = str(token_bytes)
            else:
                token_str = str(token_bytes)
            
            f.write(f"{token_id}\t{token_str}\n")
    
    print(f"Saved {len(vocab)} vocabulary entries to {filename}")

def save_merges_simple(merges, filename="merges.txt"):
    """Save merges to a simple text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, (byte1, byte2) in enumerate(merges):
            # Convert bytes to strings
            try:
                str1 = byte1.decode('utf-8')
                str2 = byte2.decode('utf-8')
            except UnicodeDecodeError:
                str1 = str(byte1)
                str2 = str(byte2)
            
            f.write(f"{i}\t{str1}\t{str2}\n")
    
    print(f"Saved {len(merges)} merges to {filename}")

# Usage with your code:
from tests.adapters import run_train_bpe

input_path = "/Users/thefoolgy/Desktop/assignment1-basics-main/data/TinyStoriesV2-GPT4-train.txt"
vocab, merges = run_train_bpe(
    input_path=input_path,
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
)

# Save to text files
save_vocab_simple(vocab, "/Users/thefoolgy/Desktop/assignment1-basics-main/data/tinystory_vocab.txt")
save_merges_simple(merges, "/Users/thefoolgy/Desktop/assignment1-basics-main/data/tinystory_merges.txt")

print("Done! Files saved:")
print("- my_vocab.txt (vocabulary)")
print("- my_merges.txt (merges)")