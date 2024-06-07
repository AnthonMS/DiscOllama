def split_code_block(codeblock, max_length=1900):
    codeblock_type = codeblock.split('\n')[0].strip()
    code = '\n'.join(codeblock.split('\n')[1:-1])  # Extract the code excluding the start and end markers
    
    if len(codeblock) <= max_length:
        return [codeblock]
    
    # Split the code into chunks
    chunks = []
    current_chunk = []
    current_length = len(codeblock_type) + 6  # `len(codeblock_type)` for the opening part + `6` for "```\n```\n"

    for line in code.split('\n'):
        if current_length + len(line) + 1 > max_length:  # +1 for the newline character
            if current_chunk:  # Only add the chunk if it's not empty
                chunks.append(f"{codeblock_type}\n{'\n'.join(current_chunk)}\n```")
            current_chunk = []
            current_length = len(codeblock_type) + 6
        
        current_chunk.append(line)
        current_length += len(line) + 1  # +1 for the newline character
    
    # Append the last chunk
    if current_chunk:
        chunks.append(f"{codeblock_type}\n{'\n'.join(current_chunk)}\n```")
    
    return chunks

# Example usage
codeblock = """```py
def test():
    # This is a test function
    pass
# Some more code
# Another line of code
# Keep adding more lines of code to exceed the 1900 characters limit...
# ...
```"""

print(codeblock)

chunks = split_code_block(codeblock, 20)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")