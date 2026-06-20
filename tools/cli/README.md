# ffwd-cli: command-line interface for ffwd

Pass one text to get its embedding, two or more to get a cosine similarity
matrix:

```bash
# one text: the embedding as a line of space-separated floats
./ffwd-cli -d ./pplx-embed-v1-0.6b "What is the capital of France?"

# two or more: a cosine similarity matrix, one row per line, no labels
./ffwd-cli -d ./pplx-embed-v1-0.6b "What is the capital of France?" \
  "Paris is the capital of France." \
  "Berlin is the capital of Germany."
# 1.000000 0.719822 0.422054
# 0.719822 1.000000 0.601722
# 0.422054 0.601722 1.000000
```

With Qwen3, prefix the retrieval query with a task instruction and embed
documents as-is. Pass them together to rank the documents against the query (the
query is the first row of the similarity matrix):

```bash
# arg 1: the query, with Qwen3's instruction prefix
# args 2+: documents
./ffwd-cli -d ./Qwen3-Embedding-0.6B \
  $'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of China?' \
  "Beijing is the capital of China." \
  "The Great Wall is a famous landmark."
```

> pplx-embed vectors are unnormalized, so rank them with cosine similarity.
> Qwen3-Embedding vectors are L2-normalized, so cosine similarity and dot
> product give the same ranking.

Pipe in lines and use `--stream` to get one JSON embedding per line:

```bash
cat texts.txt | ./ffwd-cli -d ./model --stream -b 8
# {"embedding":[...],"dim":2560,"tokens":7,"ms":12.3,"workspace_bytes":1048576}
```

Without `--stream`, reading from stdin accumulates all lines then prints the
similarity matrix.

Add `--json` to emit JSON. One text prints an array of floats; two or more print
an array of rows (the matrix):

```bash
./ffwd-cli -d ./model --json "cat" "dog"
# [[1.000000,0.380861],[0.380861,1.000000]]
```

You can configure the thread count and batch size using the `-t` and `-b`
options.

See `./ffwd-cli --help` for the full list of options.
