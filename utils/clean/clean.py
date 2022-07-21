from typing import Any
from .utils import *
from omegaconf import DictConfig
import emoji

__all__ = ["get_clean"]
__key__ = "clean"


def get_clean(clean: str) -> Any:
    clean = eval(clean)
    return clean


class GroupClean:
    def __init__(self, cfg: DictConfig) -> None:
        query = eval(cfg[__key__].query)
        self.query = query()

        bullet = eval(cfg[__key__].bullet)
        self.bullet = bullet()

        description = eval(cfg[__key__].description)
        self.description = description()

        title = eval(cfg[__key__].title)
        self.title = title()


class BaseClean:
    clean_fns = [
        "to_lower",
        "to_symbol",
        "remove_emoji",
        "clean_contractions",
        "common_us_word",
        "query_clean_v1",
        "remove_control_char",
        "remove_duplicate",
        "remove_ending_underscore",
        "remove_starting_underscore",
        "clean_multiple_form",
        "leet_clean",
    ]

    def __init__(self, clean_fns=None):
        if clean_fns:
            self.clean_fns = clean_fns

    def __call__(self, input_texts):

        if type(input_texts) == list:
            for fn in self.clean_fns:
                fn = eval(fn)
                input_texts = fn(input_texts)

        elif type(input_texts) == str:
            input_texts = [input_texts]
            input_texts = self(input_texts)
            input_texts = input_texts[0]

        return input_texts

class DeBertaCleanV2(BaseClean):
    clean_fns = [
        "to_lower",
        "to_symbol",
        "remove_emoji",
        "clean_contractions",
        "common_us_word",
        "query_clean_v1",
        "remove_control_char",
        "remove_duplicate",
        "remove_ending_underscore",
        "remove_starting_underscore",
        "clean_multiple_form",
        "leet_clean",
    ]

class DeBertaClean(BaseClean):
    clean_fns = [
        "to_symbol",
        "remove_emoji",
        "clean_contractions",
        "common_us_word",
        "query_clean_v1",
        "remove_control_char",
        "remove_duplicate",
        "remove_ending_underscore",
        "remove_starting_underscore",
        "clean_multiple_form",
        "leet_clean",
    ]


class ESclean(BaseClean):
    clean_fns = [
        "to_lower",
        "to_symbol",
        "remove_emoji",
        "clean_contractions",
        "common_es_word",
        "query_clean_v1",
        "remove_control_char",
        "remove_duplicate",
        "remove_ending_underscore",
        "remove_starting_underscore",
        "clean_multiple_form",
        "leet_clean",
    ]


def common_es_word(data):
    if type(data) == list:
        return [common_us_word(d) for d in data]
    else:
        text = data
        text = re.sub("''", '"', text)
        text = re.sub("”|“", '"', text)
        text = re.sub("‘|′", "'", text)
        exps = re.findall("[0-9] {0,1}'", text)
        for exp in exps:
            text = text.replace(exp, exp[0] + "pie")

        exps = re.findall('[0-9] {0,1}"', text)
        for exp in exps:
            text = text.replace(exp, exp.replace('"', "pulgada"))

        return text


class JSClean(BaseClean):
    clean_fns = [
        "to_lower",
        "to_symbol",
        "remove_emoji",
        "clean_contractions",
        "query_clean_v1",
        "remove_control_char",
        "remove_duplicate",
        "remove_ending_underscore",
        "remove_starting_underscore",
        "clean_multiple_form",
        "leet_clean",
    ]


def to_symbol(data):

    if type(data) == list:
        return [to_symbol(d) for d in data]
    else:
        text = data
        text = re.sub("&#34;", '"', text)
        text = re.sub("&#39;", "'", text)
        return text


def common_us_word(data):
    if type(data) == list:
        return [common_us_word(d) for d in data]
    else:
        text = data
        text = re.sub("''", '"', text)
        text = re.sub("a/c", "ac", text)
        text = re.sub("0z", "oz", text)
        text = re.sub("”|“", '"', text)
        text = re.sub("‘|′", "'", text)
        exps = re.findall("[0-9] {0,1}'", text)

        for exp in exps:
            text = text.replace(exp, exp[0] + "feet")
        exps = re.findall('[0-9] {0,1}"', text)

        for exp in exps:
            text = text.replace(exp, exp.replace('"', "inch"))

        text = re.sub("men'{0,1} {0,1}s|mens' s", "men", text)

        return text


def remove_emoji(data):
    if type(data) == list:
        return [remove_emoji(d) for d in data]
    elif type(data) == str:
        return emoji.get_emoji_regexp().sub("", data)
    else:
        raise


# TODO check spell for some words
def query_clean_v1(data):

    if type(data) == list:
        return [query_clean_v1(d) for d in data]

    elif type(data) == str:
        text = data
        product_ids = re.findall("b0[0-9a-z]{8}", text)
        if product_ids:
            for i, exp in enumerate(product_ids):
                text = text.replace(exp, f"placehold{chr(97+i)}")

        exps = re.findall("[a-zA-Z]'s|s'", text)
        for exp in exps:
            text = text.replace(exp, exp[0])

        text = re.sub("\(|\)|\*|---|\+|'|,|\[|\]| -|- |\. |/ |:", " ", text)  # ignore
        text = text.strip()

        exps = re.findall("[a-zA-Z]\.", text)
        for exp in exps:
            text = text.replace(exp, exp[0])

        # ! -> l for words
        exps = re.findall("![a-zA-Z]{2}", text)
        for exp in exps:
            text = text.replace(exp, exp.replace("!", "l"))

        # a/b -> a b
        exps = re.findall("[a-zA-Z]/[a-zA-Z]", text)
        for exp in exps:
            text = text.replace(exp, exp.replace("/", " "))

        # remove "
        text = re.sub('"', " ", text)

        # remove "
        text = re.sub("'", " ", text)

        # # + [sep] + [num] -> # + [num]
        exps = re.findall("# {1}[0-9]", text)
        for exp in exps:
            text = text.replace(exp, exp.replace(" ", ""))

        # remove # without
        exps = re.findall("#[a-zA-Z]", text)
        for exp in exps:
            text = text.replace(exp, exp.replace("#", ""))

        if product_ids:
            for i, exp in enumerate(product_ids):
                text = text.replace(f"placehold{chr(97+i)}", exp)

        text = text.strip()

        return text
