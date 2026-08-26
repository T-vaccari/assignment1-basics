from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from multiprocessing import Pool







def process_chunk(start,end,segment_delimiter,PAT,input_path):
   with open(input_path, "rb") as f:
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
   
   local_pre_token_count = dict()

   for doc in docs:
   
      for match in re.finditer(PAT, doc): # Process every pre-token finded by the regex
         match_encoded = match.group(0).encode("utf-8")
         tuple_of_bytes = tuple([match_encoded[i:i+1] for i in range(len(match_encoded)) ])
         local_pre_token_count[tuple_of_bytes] = local_pre_token_count.get(tuple_of_bytes, 0) + 1

   return local_pre_token_count








def train_bpe(
input_path: str,
vocab_size: int,
special_tokens: list[str]
   
)-> tuple[dict[int,bytes], list[tuple[bytes,bytes]]]:

   # Regex pattern 
   PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

   pre_token_count = dict()

   vocab = {i: bytes([i]) for i in range(256)}

   merges = list()

   ## Usage
   with open(input_path, "rb") as f:
      num_processes = 4
      boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

   # The following is a serial implementation, but you can parallelize this
   # by sending each start/end pair to a set of processes.

   escaped_special_tokens = [re.escape(st) for st in special_tokens]
   segment_delimiter = "|".join(escaped_special_tokens)


   jobs = []
   for start, end in zip(boundaries[:-1], boundaries[1:]):
      # Launch process indipendently
      # then wait for them to finish and then merge into a unique vocab
      jobs.append((start,end,segment_delimiter,PAT,input_path))
      
   pool = Pool(num_processes)
   results = pool.starmap(process_chunk, jobs)

   # Now i have to merge the result into a unique pre_token_count

   pre_token_count = dict()

   for lptc in results:
      for pre_token in lptc.keys():
         pre_token_count[pre_token] = lptc.get(pre_token, 0) + pre_token_count.get(pre_token,0)

      


   while len(vocab) < (vocab_size - len(special_tokens) ):

      
      # Now i have the pre-token dictionary, that takes into account of every chunk, now I can start to count the byte pairs and simply multiply
      # by the occurences of that precise pre-token

      pair_count: dict[tuple[bytes, bytes], int] = {}

      for pre_token in pre_token_count.keys(): #Iterate along all the keys
         for i in range(len(pre_token)-1):
   
            a = pre_token[i]
            b = pre_token[i+1]
            pair = (a,b)
            # print(pre_token, len(pre_token),type(pre_token),print(pre_token[i:i+1]),tuple([pre_token[i],pre_token[i+1]]),pair)
            pair_count[pair]  = pair_count.get(pair, 0) + 1 * pre_token_count[pre_token]
   


      # print(pair_count)
      # Now I can pick-up the most frequent pair, if we have a tie I must choose the lexicographically greater pair
      
      # To obtain the top-pair I can use the lambda that creates tuples, if we have a tie on the appearences then python compares lexicographically 
      # the byte pairs
      if len(pair_count) == 0:
         break
      top_pair = max(pair_count, key = lambda  x : (pair_count[x], x))
      # print(top_pair,type(top_pair))

      # Now I can append the top sequences in the merges, and then I have to substitue in each occurence of the pair with the new byte
      merges.append(top_pair)

      # Now that I have the top pair I can start the merging process, I need to add to the merges the pair chosen for the process
      # When i find the pair, i substitute the occurence in the tuple with the new b'pair' and i can assign to them a new token id

      vocab[len(vocab)] = top_pair[0] + top_pair[1]

      # Now I have to substitute each occuren of the top pair with in the tuple with the fused byte sequence

      new_pre_token_count = dict()

      for pre_token in pre_token_count.keys(): #Iterate along all the keys
         # I have to modify the keys, so now I find that a key has the top_pair in it, I must substitue the tuple, removing
         # the old separate entries for a and b and obtaining (old bytes*, b'top_pair', old_bytes*)
         merged_pre_token = list()
         i = 0
         while i < len(pre_token):
            if i < len(pre_token) -1 and top_pair[0]+ top_pair[1] == pre_token[i] + pre_token[i+1]:
               merged_pre_token.append(top_pair[0] + top_pair[1])
               i+=2
            else:
               merged_pre_token.append(pre_token[i])
               i+=1

         merged_pre_token = tuple(merged_pre_token)
         # print(pre_token,merged_pre_token,top_pair)
         new_pre_token_count[merged_pre_token] = pre_token_count[pre_token]

      pre_token_count = new_pre_token_count


      # Now I have to recompute stats for the next merge round, and repeat this process until the vocab size length desired is not reached

   # At the end I can insert in the vocab the special tokens
   for st in special_tokens:
      vocab[len(vocab)] = st.encode("utf-8")  

               


   return (vocab, merges)




            


         
               




vocab , merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 150, ["<|endoftext|>"])
# vocab , merges = train_bpe("data/test.txt", 258, ["<|endoftext|>"])
print(vocab, merges)