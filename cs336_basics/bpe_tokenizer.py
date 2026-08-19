from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
def train_bpe(
input_path: str,
vocab_size: int,
special_tokens: list[str]
   
)-> tuple[dict[int,bytes], list[tuple[bytes,bytes]]]:

   # Regex pattern 
   PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

   pre_token_count = dict()

   ## Usage
   with open(input_path, "rb") as f:
      num_processes = 4
      boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

      # The following is a serial implementation, but you can parallelize this
      # by sending each start/end pair to a set of processes.

      escaped_special_tokens = [re.escape(st) for st in special_tokens]
      segment_delimiter = "|".join(escaped_special_tokens)




      for start, end in zip(boundaries[:-1], boundaries[1:]):
         f.seek(start)
         chunk = f.read(end - start).decode("utf-8", errors="ignore")
         # Run pre-tokenization on your chunk and store the counts for each pre-token
         """
         The steps to perform in order are the following:
         1)Chunk safealy to obtain chunk to process in parallel(still containing delimiters)
         2)For each chunk strip out all special tokens, and obtain documents from the delimiter, we want to pre-tokenize 
            separately doc1 and doc2 where we have doc1 <|endoftext|>
         3)Run the regex pre tokenization on each doc indipendently in each worker and then I can accumulate per token count.
         
         """

         #Split in docs
         

         docs = re.split(segment_delimiter,chunk)

         #For each docs separately I need to tokenize 

         for doc in docs:

            for match in re.finditer(PAT, doc):
               match_encoded = match.group(0).encode("utf-8")
               pre_token_count[bytes(match_encoded)] = pre_token_count.get(tuple(match_encoded), 0) + 1
               

   print(pre_token_count.keys(), pre_token_count.values())

   for l in pre_token_count.keys():
      print(bytes(l).decode("utf-8"))



            


         
               




# train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 150, ["<|endoftext|>"])
train_bpe("data/test.txt", 150, ["<|endoftext|>"])