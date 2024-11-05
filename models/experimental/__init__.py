
from models.experimental.byte_level import (
    ByteModelShell,
    ByteLevelEmbedder,
    ByteLevelDecoder
)

EXPERIMENTAL_EMBEDDING_MODEL_DICT = {
    "byte_level_embedder": ByteLevelEmbedder
}

EXPERIMENTAL_CORE_MODEL_DICT = {
}

EXPERIMENTAL_MODEL_HEAD_DICT = {
    "byte_level_head": ByteLevelDecoder
}

EXPERIMENTAL_MODEL_SHELL_DICT = {
    "byte_model_shell": ByteModelShell
}