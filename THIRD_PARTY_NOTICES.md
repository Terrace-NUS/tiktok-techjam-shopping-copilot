# Third-Party Notices

This project depends on third-party packages declared in `pyproject.toml`.
Their dependency distributions retain their own license texts and notices.

## RFC 8785 canonical JSON

`rfc8785` is maintained by Trail of Bits and distributed under the Apache
License 2.0. The project uses it for JSON Canonicalization Scheme encoding.

- Project: <https://github.com/trailofbits/rfc8785.py>
- License: <https://github.com/trailofbits/rfc8785.py/blob/master/LICENSE>

## Unicode normalization

`pyunormalize` is maintained by Marc Lodewijck and distributed under the MIT
License. It also carries the Unicode data terms applicable to its generated
normalization tables.

- Project: <https://github.com/mlodewijck/pyunormalize>
- Package: <https://pypi.org/project/pyunormalize/17.0.0/>

## Unicode case folding data

`src/shopping_copilot/catalog/semantic/category/_casefold_v17.py` is generated
from the Unicode Consortium's `CaseFolding-17.0.0.txt`. The generator verifies
the exact upstream SHA-256 before accepting the input. The derived table is
distributed under the Unicode License V3 reproduced with this repository and
installed distribution.

- Source: <https://www.unicode.org/Public/17.0.0/ucd/CaseFolding.txt>
- License: [`licenses/UNICODE-3.0.txt`](licenses/UNICODE-3.0.txt)
- Source SHA-256: `ff8d8fefbf123574205085d6714c36149eb946d717a0c585c27f0f4ef58c4183`

## BGE small English embedding model

The optional dense retrieval workflow downloads `BAAI/bge-small-en-v1.5` at
the immutable revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
The model is distributed under the MIT License. Model weights are cached
locally and are not committed to this repository.

- Model card: <https://huggingface.co/BAAI/bge-small-en-v1.5>
- Pinned revision: <https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a>

The optional retrieval dependency stack includes NumPy, Sentence Transformers,
Transformers, and PyTorch. Their distributions retain their own license files
and notices; exact package versions and hashes are recorded in
`requirements/retrieval.lock`.
