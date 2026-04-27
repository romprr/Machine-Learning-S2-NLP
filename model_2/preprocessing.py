import argparse
import html
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Iterable
from urllib.parse import unquote

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

try:
    from .pipeline_paths import PREPROCESSED_DATA_PATH, PREPROCESSING_SUMMARY_PATH, RAW_DATA_PATH, ensure_pipeline_directories
except ImportError:
    from pipeline_paths import PREPROCESSED_DATA_PATH, PREPROCESSING_SUMMARY_PATH, RAW_DATA_PATH, ensure_pipeline_directories


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
PERCENT_ENCODING_PATTERN = re.compile(r"%[0-9a-f]{2}", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<\s*/?\s*([a-z][a-z0-9-]*)\b([^>]*)>", re.IGNORECASE)
CODE_ALLOWED_PATTERN = re.compile(r"[^a-z0-9\s._#+\-<>\[\](){}/\\=,:;*^%!?&|~]")

HTML_SIGNAL_TAGS = {
    "a",
    "body",
    "blockquote",
    "button",
    "code",
    "div",
    "dl",
    "dt",
    "dd",
    "em",
    "form",
    "h1",
    "h2",
    "h3",
    "head",
    "hr",
    "html",
    "iframe",
    "img",
    "input",
    "label",
    "li",
    "link",
    "meta",
    "ol",
    "option",
    "p",
    "script",
    "select",
    "span",
    "style",
    "table",
    "tbody",
    "td",
    "textarea",
    "th",
    "thead",
    "title",
    "tr",
    "ul",
} - {"code"}
HTML_SIGNAL_ATTRIBUTES = {
    "action",
    "class",
    "href",
    "id",
    "method",
    "name",
    "rel",
    "src",
    "style",
    "type",
    "value",
}

# Keep the 20 dataset labels intact so punctuation cleanup does not destroy them.
TECHNOLOGY_LABELS = [
    ".net",
    "android",
    "angularjs",
    "asp.net",
    "c",
    "c#",
    "c++",
    "css",
    "html",
    "ios",
    "iphone",
    "java",
    "javascript",
    "jquery",
    "mysql",
    "objective-c",
    "php",
    "python",
    "ruby-on-rails",
    "sql",
]

TECH_TOKEN_MAP = {
    label: f"techlabelx{index:02d}x"
    for index, label in enumerate(sorted(TECHNOLOGY_LABELS, key=len, reverse=True), start=1)
}
TECH_TOKEN_RESTORE = {placeholder: label for label, placeholder in TECH_TOKEN_MAP.items()}
CUSTOM_STOP_WORDS = {"given", "here"}

# Keep method-like tokens, array types, and regex/code patterns such as [^\w\s] and -\d+.
TOKEN_PATTERN = (
    r"(?u)\[[^\]\s]+\]|-?\\[a-z]+\+?|"
    r"\b[a-z0-9_]+(?:\[\])?(?:[.#+\-][a-z0-9_]+(?:\[\])?)*\b|c\+\+|c#|\.net"
)

# Keep a few meaningful markers even when removing stopwords.
PROTECTED_TOKENS = {
    "code_block",
    "code_block_start",
    "code_block_end",
    "inline_code",
    "link",
    "quote_block",
    "list_item",
    "url",
    *TECH_TOKEN_RESTORE.keys(),
}


def extract_html_signal_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for tag_name, attr_text in HTML_TAG_PATTERN.findall(text):
        normalized_tag = tag_name.lower()
        if normalized_tag in HTML_SIGNAL_TAGS:
            tokens.append(f"html_tag_{normalized_tag}")

        lowered_attr_text = attr_text.lower()
        for attr_name in HTML_SIGNAL_ATTRIBUTES:
            if re.search(rf"\b{attr_name}\s*=", lowered_attr_text):
                tokens.append(f"html_attr_{attr_name}")

    return list(dict.fromkeys(tokens))


def protect_technical_tokens(text: str) -> str:
    protected = text
    for label, placeholder in TECH_TOKEN_MAP.items():
        protected = re.sub(rf"(?<!\w){re.escape(label)}(?!\w)", f" {placeholder} ", protected)
    return protected


def restore_technical_tokens(text: str) -> str:
    restored = text
    for placeholder, label in sorted(TECH_TOKEN_RESTORE.items(), key=lambda item: len(item[0]), reverse=True):
        restored = restored.replace(placeholder, label)
    return restored


def replace_urls(text: str) -> str:
    return URL_PATTERN.sub(" url ", text)


def split_code_statements(text: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    paren_depth = 0

    for char in text:
        if char == "(":
            paren_depth += 1
            current.append(char)
            continue

        if char == ")":
            paren_depth = max(paren_depth - 1, 0)
            current.append(char)
            continue

        if char in "{}":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            paren_depth = 0
            continue

        if char == ";" and paren_depth == 0:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            continue

        current.append(char)

    statement = "".join(current).strip()
    if statement:
        statements.append(statement)

    return statements


def clean_code_line(text: str, preserve_technical_tokens: bool = True) -> str:
    cleaned = html.unescape(unquote(str(text))).lower()
    cleaned = cleaned.replace("&lt;", "<").replace("&gt;", ">")
    cleaned = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", cleaned)
    cleaned = re.sub(r"-\s+(\\[a-z]+\+?)", r"-\1", cleaned)

    if preserve_technical_tokens:
        cleaned = protect_technical_tokens(cleaned)

    cleaned = CODE_ALLOWED_PATTERN.sub(" ", cleaned)
    cleaned = normalize_spaces(cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"\[\s+", "[", cleaned)
    cleaned = re.sub(r"\s+\]", "]", cleaned)
    cleaned = re.sub(r"<\s+", "<", cleaned)
    cleaned = re.sub(r"\s+>", ">", cleaned)
    cleaned = re.sub(r"(?<=[a-z0-9_])\s+\[\]", "[]", cleaned)
    cleaned = re.sub(r"\s*\.\s*", ".", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"(?<![+\-*/%&|^!<>])\s*=\s*(?![=])", " = ", cleaned)
    cleaned = normalize_spaces(cleaned)

    if preserve_technical_tokens:
        cleaned = restore_technical_tokens(cleaned)
        cleaned = normalize_spaces(cleaned)

    return cleaned


def clean_code_block(text: str, preserve_technical_tokens: bool = True) -> str:
    normalized = html.unescape(unquote(str(text))).lower()
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("&lt;", "<").replace("&gt;", ">")

    # Recover line structure without splitting for-loop headers on semicolons.
    normalized_lines: list[str] = []
    for raw_segment in normalized.splitlines():
        normalized_lines.extend(split_code_statements(raw_segment))

    normalized = "\n".join(normalized_lines)
    normalized = re.sub(r"\)\s*(?=(?:catch|else|for|if|while)\b)", ")\n", normalized)
    normalized = re.sub(r"\b(?:catch|else|finally|for|if|while)\b", lambda m: f"\n{m.group(0)}", normalized)

    cleaned_lines = []
    for raw_line in normalized.splitlines():
        cleaned_line = clean_code_line(raw_line, preserve_technical_tokens=preserve_technical_tokens)
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)


def handle_html_structure(text: str, keep_code_content: bool, preserve_technical_tokens: bool) -> tuple[str, list[tuple[str, str]]]:
    soup = BeautifulSoup(text, "html.parser")
    code_blocks: list[tuple[str, str]] = []

    for index, pre_tag in enumerate(soup.find_all("pre")):
        pre_text = pre_tag.get_text("\n", strip=True)
        placeholder = f"codeblockplaceholderx{index}x"
        cleaned_code = clean_code_block(pre_text, preserve_technical_tokens=preserve_technical_tokens)
        replacement_text = f" {placeholder} " if keep_code_content else " code_block "
        code_blocks.append((placeholder, cleaned_code))
        pre_tag.replace_with(replacement_text)

    for anchor_tag in soup.find_all("a"):
        link_text = anchor_tag.get_text(" ", strip=True)
        href = anchor_tag.get("href", "")
        has_href = bool(href.strip())
        replacement_parts = [" link "]
        if link_text:
            replacement_parts.append(link_text)
        if has_href:
            replacement_parts.append(" url ")
        anchor_tag.replace_with(" ".join(replacement_parts))

    for blockquote_tag in soup.find_all("blockquote"):
        quote_text = blockquote_tag.get_text(" ", strip=True)
        blockquote_tag.replace_with(f" quote_block {quote_text} ")

    for list_item_tag in soup.find_all("li"):
        item_text = list_item_tag.get_text(" ", strip=True)
        list_item_tag.replace_with(f" list_item {item_text} ")

    for code_tag in soup.find_all("code"):
        code_text = clean_code_line(code_tag.get_text(" ", strip=True), preserve_technical_tokens=preserve_technical_tokens)
        replacement_text = f" inline_code {code_text} " if code_text else " inline_code "
        code_tag.replace_with(replacement_text)

    return soup.get_text(" ", strip=False), code_blocks


def remove_noise_punctuation(text: str) -> str:
    cleaned = text
    cleaned = cleaned.replace("{{", " ").replace("}}", " ")
    cleaned = cleaned.replace("{%", " ").replace("%}", " ")
    cleaned = cleaned.replace("&lt;", " ").replace("&gt;", " ")

    # Remove lingering percent-encoded fragments that survive URL decoding.
    cleaned = PERCENT_ENCODING_PATTERN.sub(" ", cleaned)

    # Keep letters, digits, whitespace and punctuation that carries technical signal.
    cleaned = re.sub(r"[^a-z0-9\s._#+\-]", " ", cleaned)
    cleaned = re.sub(r"[_]{2,}", "_", cleaned)
    cleaned = re.sub(r"[.]{2,}", ".", cleaned)
    cleaned = re.sub(r"[-]{2,}", "-", cleaned)
    return cleaned


def remove_stopwords(text: str) -> str:
    tokens = re.findall(TOKEN_PATTERN, text)
    filtered_tokens = [
        token
        for token in tokens
        if token in PROTECTED_TOKENS or token not in ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS)
    ]
    return " ".join(filtered_tokens)


def normalize_spaces(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def preprocess_text(
    text: str,
    keep_code_content: bool = True,
    replace_url_tokens: bool = True,
    preserve_technical_tokens: bool = True,
    remove_stop_words: bool = True,
) -> str:
    normalized = str(text)
    normalized = unquote(normalized)
    normalized = normalized.lower()
    normalized, code_blocks = handle_html_structure(
        normalized,
        keep_code_content=keep_code_content,
        preserve_technical_tokens=preserve_technical_tokens,
    )

    if preserve_technical_tokens:
        normalized = protect_technical_tokens(normalized)

    if replace_url_tokens:
        normalized = replace_urls(normalized)

    normalized = remove_noise_punctuation(normalized)
    normalized = normalize_spaces(normalized)

    if remove_stop_words:
        normalized = remove_stopwords(normalized)
        normalized = normalize_spaces(normalized)

    if preserve_technical_tokens:
        normalized = restore_technical_tokens(normalized)
        normalized = normalize_spaces(normalized)

    if keep_code_content:
        for placeholder, cleaned_code in code_blocks:
            replacement_text = (
                f"\ncode_block\n{cleaned_code}\ncode_block_end\n"
                if cleaned_code
                else "\ncode_block\ncode_block_end\n"
            )
            normalized = normalized.replace(placeholder, replacement_text)

        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()

    return normalized


def preprocess_keep_code(text: str, remove_stop_words: bool = True) -> str:
    return preprocess_text(text, keep_code_content=True, remove_stop_words=remove_stop_words)


def preprocess_code_block_token(text: str, remove_stop_words: bool = True) -> str:
    return preprocess_text(text, keep_code_content=False, remove_stop_words=remove_stop_words)


def preprocess_series(
    posts: Iterable[str],
    keep_code_content: bool = True,
    remove_stop_words: bool = True,
) -> pd.Series:
    function = preprocess_keep_code if keep_code_content else preprocess_code_block_token
    return pd.Series(posts).astype(str).apply(lambda text: function(text, remove_stop_words))


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        preprocessor=None,
        tokenizer=None,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def build_char_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def preprocess_dataframe(
    df: pd.DataFrame,
    keep_code_content: bool = True,
    remove_stop_words: bool = True,
) -> pd.DataFrame:
    required_columns = {"post", "tags"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_text}")

    processed = df.copy()
    if "row_id" not in processed.columns:
        processed.insert(0, "row_id", range(len(processed)))

    processed["post"] = processed["post"].astype(str)
    processed["tags"] = processed["tags"].astype(str)
    processed["preprocessed_post"] = preprocess_series(
        processed["post"],
        keep_code_content=keep_code_content,
        remove_stop_words=remove_stop_words,
    )
    return processed


def summarize_preprocessed_dataframe(
    df: pd.DataFrame,
    keep_code_content: bool,
    remove_stop_words: bool,
    input_path: Path,
    output_path: Path,
    duration_seconds: float,
) -> dict[str, object]:
    preprocessed_posts = df["preprocessed_post"].astype(str)

    return {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "keep_code_content": keep_code_content,
        "remove_stop_words": remove_stop_words,
        "unique_tags": int(df["tags"].nunique()),
        "code_block_rows": int(preprocessed_posts.str.contains(r"\bcode_block\b", regex=True).sum()),
        "average_preprocessed_characters": float(preprocessed_posts.str.len().mean()),
        "average_preprocessed_tokens": float(preprocessed_posts.str.split().str.len().mean()),
        "duration_seconds": round(duration_seconds, 4),
    }


def preprocess_dataset(
    input_path: Path | str = RAW_DATA_PATH,
    output_path: Path | str = PREPROCESSED_DATA_PATH,
    summary_path: Path | str = PREPROCESSING_SUMMARY_PATH,
    keep_code_content: bool = True,
    remove_stop_words: bool = True,
) -> tuple[Path, Path]:
    ensure_pipeline_directories()

    input_path = Path(input_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find dataset: {input_path}")

    started_at = perf_counter()
    dataframe = pd.read_csv(input_path)
    processed = preprocess_dataframe(
        dataframe,
        keep_code_content=keep_code_content,
        remove_stop_words=remove_stop_words,
    )
    duration_seconds = perf_counter() - started_at

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)

    summary = summarize_preprocessed_dataframe(
        processed,
        keep_code_content=keep_code_content,
        remove_stop_words=remove_stop_words,
        input_path=input_path,
        output_path=output_path,
        duration_seconds=duration_seconds,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path, summary_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess the Stack Overflow dataset and save a CSV with a preprocessed_post column."
    )
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH, help="Path to the raw CSV dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PREPROCESSED_DATA_PATH,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PREPROCESSING_SUMMARY_PATH,
        help="Path to the JSON summary file.",
    )
    parser.add_argument(
        "--code-block-token-only",
        action="store_true",
        help="Replace each <pre> block with the token code_block instead of keeping cleaned code content.",
    )
    parser.add_argument(
        "--keep-stop-words",
        action="store_true",
        help="Keep stop words in the preprocessed output.",
    )
    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    arguments = parser.parse_args()

    output_path, summary_path = preprocess_dataset(
        input_path=arguments.input,
        output_path=arguments.output,
        summary_path=arguments.summary_output,
        keep_code_content=not arguments.code_block_token_only,
        remove_stop_words=not arguments.keep_stop_words,
    )
    print(f"Saved preprocessed dataset to: {output_path.resolve()}")
    print(f"Saved preprocessing summary to: {summary_path.resolve()}")
