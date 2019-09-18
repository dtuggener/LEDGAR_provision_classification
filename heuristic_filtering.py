import re
from typing import List, Union, Set


def process_label(label: str, lowercase: bool = True, stop_words: Set[str] = None) -> Union[List[str], None]:
    """Heuristically filter and process label(s)"""

    if not label:
        return None

    # Handle multi-labels
    label_delimiters_regex = re.compile('|'.join([';', '/']))
    labels = set(l.strip() for l in re.split(label_delimiters_regex, label))

    blacklist = ['section', 'etc', 'now', 'whereas', 'exhibit ',
                 'therefore', 'article', 'in witness whereof', 'schedule', 'article']
    filtered_labels = set([])

    for label in labels:

        if len(label) < 3 or len(label) > 75 or  \
                 not label[0].isupper() or  \
                any(bw for bw in blacklist if label.lower().startswith(bw)):
            continue

        if label[-1] in ['.', ':']:  # remove scraping artifacts
            label = label[:-1]

        label = re.sub('[ \t]+', ' ', label.replace('\n', ' ').strip())

        if label:

            if stop_words:
                if label.lower() in stop_words:
                    continue
                label_words = label.split(' ')
                if len(label_words) > 1 and len(label_words[-1]) > 1 \
                        and label_words[-1] in stop_words:
                    continue

            label = label.lower() if lowercase else label
            filtered_labels.add(label)

    return list(filtered_labels)


def process_text(text: str) -> Union[str, None]:
    """Heuristically filter and process provision text"""

    text = text.strip()

    blacklist = ["” means", '" means', 'shall mean', "' means", '’ means'
                 'shall have the meaning', 'has the meaning', 'have meaning']

    if len(text) < 25 or \
            text[0].islower() or \
            text[0] in ['"', '”'] or \
            any(bw for bw in blacklist if bw in text):
        return None

    text = text.strip()
    if text[0] in ['.', ':']:
        text = text[1:].strip()

    if not text[0].isupper() and not text[0] in ['(', '[']:
        return None

    if not text[-1] == '.':
        return None

    text = re.sub('[ \t]+', ' ', text.replace('\n', ' ').strip())

    return text
