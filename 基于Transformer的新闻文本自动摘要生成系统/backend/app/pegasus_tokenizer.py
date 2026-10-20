"""
Randeng-Pegasus 中文摘要模型的自定义分词器。

来源：IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese 仓库官方提供的
tokenizers_pegasus.py。该模型未发布标准 tokenizer 文件（无 spiece.model /
tokenizer.json），仅提供 vocab.txt（BERT WordPiece 词表）与本源文件，
因此无法通过 AutoTokenizer 直接加载，必须使用本模块手动实例化。

分词流程：jieba 预分词 → BERT BasicTokenizer（中文按字拆分）→ WordPiece。
"""
import collections
import os
import re
import unicodedata
from typing import List, Optional, Tuple, Union

import jieba
from transformers import PreTrainedTokenizer
from transformers import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# jieba 临时目录
jieba.dt.tmp_dir = os.path.expanduser("~/.cache/")
jieba.initialize()


# ---------------------------------------------------------------------------
# 以下 4 个字符判断函数来自官方 data_utils.py（标准 BERT 实现）
# ---------------------------------------------------------------------------
def _is_chinese_char(cp):
    if ((cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)):
        return True
    return False


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    return text.split()


class PegasusTokenizer(PreTrainedTokenizer):
    """Randeng-Pegasus 中文分词器（WordPiece + jieba）。"""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 never_split=None,
                 pad_token="<pad>",
                 eos_token="</s>",
                 unk_token="<unk>",
                 mask_token="<mask_2>",
                 mask_token_sent="<mask_1>",
                 additional_special_tokens=None,
                 sep_token="[SEP]",
                 cls_token="[CLS]",
                 tokenize_chinese_chars=True,
                 strip_accents=None,
                 offset=100,
                 pre_tokenizer=lambda x: jieba.cut(x, HMM=False),
                 **kwargs):
        self.offset = offset

        if additional_special_tokens is not None:
            if not isinstance(additional_special_tokens, list):
                raise TypeError(
                    f"additional_special_tokens should be of type {type(list)}, "
                    f"but is {type(additional_special_tokens)}"
                )

            additional_special_tokens_extended = (
                ([mask_token_sent] + additional_special_tokens)
                if mask_token_sent not in additional_special_tokens
                and mask_token_sent is not None else additional_special_tokens)

            # fill additional tokens with ..., <unk_token_102> in case not all
            # additional tokens are already taken
            additional_special_tokens_extended += [
                f"<unk_{i}>" for i in range(
                    len(additional_special_tokens_extended), self.offset - 1)
            ]

            if len(set(additional_special_tokens_extended)) != len(
                    additional_special_tokens_extended):
                raise ValueError(
                    "Please make sure that the provided additional_special_tokens "
                    "do not contain an incorrectly shifted list of <unk_x> tokens. "
                    f"Found {additional_special_tokens_extended}."
                )
            additional_special_tokens = additional_special_tokens_extended
        else:
            additional_special_tokens = [
                mask_token_sent
            ] if mask_token_sent is not None else []

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'."
            )

        # 先加载词表并把特殊 token 放入 BERT 词表保留位，
        # 这样基类 __init__ 注册特殊 token 时会复用保留位 id（0/1/2/3/100），
        # 而不是追加到词表末尾（50000+），与模型权重中的 embedding 对齐。
        self.offset = offset
        self.pre_tokenizer = pre_tokenizer
        self.mask_token_sent = mask_token_sent
        self.vocab = load_vocab(vocab_file)
        self.vocab[eos_token] = self.vocab.pop("[unused1]")   # id 1
        self.vocab[pad_token] = self.vocab.pop("[PAD]")       # id 0
        self.vocab[unk_token] = self.vocab.pop("[UNK]")       # id 100
        if mask_token_sent is not None:
            self.vocab[mask_token] = self.vocab.pop("[unused3]")      # id 3
            self.vocab[mask_token_sent] = self.vocab.pop("[unused2]")  # id 2
        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])

        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            eos_token=eos_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            additional_special_tokens=additional_special_tokens,
            strip_accents=strip_accents,
            **kwargs,
        )

        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=self.unk_token)

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                if self.do_basic_tokenize:
                    for token in self.basic_tokenizer.tokenize(
                            text, never_split=self.all_special_tokens):
                        if token in self.basic_tokenizer.never_split:
                            split_tokens.append(token)
                        else:
                            split_tokens += self.wordpiece_tokenizer.tokenize(token)
                else:
                    split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\
            \uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\
            \uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\
            \u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\
            \u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'

    def convert_ids_to_tokens(self, ids, skip_special_tokens: bool = False):
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids and index != 2:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def convert_tokens_to_string(self, tokens):
        text = ''
        for i, token in enumerate(tokens):
            if token[:2] == '##':
                text += token[2:]
            elif len(token) == 1 and _is_chinese_char(ord(token)):
                text += token
            elif len(token) == 1 and _is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and _is_chinese_char(ord(text[-1])):
                text += token
            elif tokens == "</s>":
                continue
            else:
                text += ' '
                text += token
        text = re.sub(' +', ' ', text)
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation = re.sub(' +', '', self._cjk_punctuation()).strip() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub(r'(\d\.) (\d)', '\\1\\2', text)
        return text.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [self.eos_token_id]
        else:
            return self._special_token_mask(token_ids_0 +
                                            token_ids_1) + [self.eos_token_id]

    def num_special_tokens_to_add(self, pair=False):
        return 1

    def save_vocabulary(self, save_directory: str,
                        filename_prefix: Optional[str] = None) -> Tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") +
                VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = ((filename_prefix + "-" if filename_prefix else "")
                          + save_directory)
        with open(vocab_file, "w", encoding="utf-8") as writer:
            index = 0
            for token, token_index in sorted(self.vocab.items(),
                                             key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices "
                        "are not consecutive.")
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class BasicTokenizer(object):
    """BERT 风格基础分词器（清洗、中文按字拆分、标点切分）。"""

    def __init__(self,
                 do_lower_case=True,
                 never_split=None,
                 tokenize_chinese_chars=True,
                 strip_accents=None):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        never_split = self.never_split.union(
            set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))
        return whitespace_tokenize(" ".join(split_tokens))

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            if unicodedata.category(char) == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        return _is_chinese_char(cp)

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            output.append(" " if _is_whitespace(char) else char)
        return "".join(output)


class WordpieceTokenizer(object):
    """WordPiece 最长匹配子词切分。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
