"""
sec.gov scaper that extracts provisions and their labels from exhibit 10* filings downloaded with the sec_crawler.py
"""

import re
import os
import glob
import json
import random; random.seed(42)
from html import unescape
from typing import List, Tuple, Set
from dataclasses import dataclass
from nltk.corpus import stopwords

from heuristic_filtering import process_text, process_label


@dataclass
class LabeledProvision:
    provision: str
    label: List[str]
    source: str


def scrape_u_tag(p: str) -> Tuple[str, str]:
    label, text = None, None
    underline = re.search('<[Uu]>', p)
    if underline:
        underline_tag = underline.group()
        underline_end_tag = underline_tag.replace('<', '</')
        label = re.search(underline_tag + '(.*?)' + underline_end_tag, p, re.S)
        text = re.search(underline_end_tag + '(.*)', p, re.S)

        if label and text:
            label = label.group(1).strip()
            text = text.group(1).strip()
            # Remove html markup
            label = re.sub('<[^>]+>', ' ', label).replace('\n', ' ').strip()
            text = re.sub('<[^>]+>', ' ', text).replace('\n', ' ').strip()

    return label, text


def scrape_font_tag(p: str) -> Tuple[str, str]:
    """Extract label and text from paragraph block which contains <font> tag"""
    label, text = None, None
    fonts = re.findall('(<font style=[^>]+>)(.*?)</font>', p, re.S)
    if len(fonts) > 1:  # Only one element means there isn't different formatting to indicate a label
        
        # Check for different font styles
        if len(set([f[0] for f in fonts])) > 1:

            # Check for underline formatting
            font_underline, font_bold = False, False
            if 'underline' in p:
                font_underline = True
            elif 'bold' in p:
                font_bold = True

            label, text = [], []
            label_detected, label_finished = False, False
            for f in fonts:
                # Consume font tags until label markup is no longer detected
                if not label_finished and not label_detected:

                    if font_underline and 'underline' in f[0]:
                        label.append(f[1])
                        label_detected = True

                    elif font_bold and 'bold' in f[0]:
                        label.append(f[1])
                        label_detected = True

                elif label_detected:
                    text.append(f[1])
                    label_finished = True

            if text and label:
                text = ' '.join(text).strip()
                label = ' '.join(label).strip()
                label = re.sub('<[^>]+>', ' ', label).replace('\n', ' ').strip()
                text = re.sub('<[^>]+>', ' ', text).replace('\n', ' ').strip()

    return label, text


def scrape_exhibit_10(html_file: str, filtering: bool = True, stop_words: Set[str] = None) -> List[LabeledProvision]:
    """Parse exhibit 10 htm files"""

    html = open(html_file).read()

    # Two major html layouts: <p> or <div> tags for paragraphs
    if '<p' in html or '<P' in html:
        elem_regex = re.compile('<[Pp][^>]*>(.*?)</[Pp]>', re.S)
    else:
        elem_regex = re.compile('<div[ >].*?</div>', re.S)

    # Label highlighting is either <u> or <font> tag
    u_tag, font_tag = False, False
    if '<u>' in html or '<U>' in html:
        u_tag = True
    elif '<font' in html and ('underline' in html or 'bold' in html):
        font_tag = True

    ps = elem_regex.findall(html)
    provisions_doc: List[LabeledProvision] = []
    for p in ps:
        p = unescape(p).strip()

        if u_tag:
            label, text = scrape_u_tag(p)
        elif font_tag:
            label, text = scrape_font_tag(p)
        else:
            label, text = None, None

        if label and text:

            if filtering:
                labels = process_label(label, stop_words=stop_words)
                text = process_text(text)
            else:
                labels = [label]

            if labels and text:
                contract_uri = '/'.join(html_file.split('/')[-4:])
                labeled_provision = LabeledProvision(text, labels, contract_uri)
                provisions_doc.append(labeled_provision)

    return provisions_doc


def scrape_by_year(data_dir: str, years: range = range(2019, 1992, -1),
                   qs=None, max_contracts=-1, verbose: bool = True,
                   filtering: bool = True, stop_words: Set[str] = None) -> List[LabeledProvision]:

    if qs is None:
        qs = ['QTR1', 'QTR2', 'QTR3', 'QTR4']

    contracts_scraped = 0
    provisions: List[LabeledProvision] = []

    for year in years:
        year = str(year)
        year_dir = os.path.join(data_dir, year)
        if not os.path.exists(year_dir):
            continue

        for q in qs:
            year_q_dir = os.path.join(year_dir, q)
            if not os.path.exists(year_q_dir):
                continue

            for folder in os.listdir(year_q_dir):
                if not os.path.isdir(os.path.join(year_q_dir, folder)):
                    continue

                for fname in os.listdir(os.path.join(year_q_dir, folder)):

                    if fname.endswith('.htm'):
                        html_file = os.path.join(year_q_dir, folder, fname)
                        if verbose:
                            print('Scraping', contracts_scraped, html_file)
                        provisions_doc = scrape_exhibit_10(html_file, filtering=filtering, stop_words=stop_words)

                        if provisions_doc:
                            provisions.extend(provisions_doc)

                            contracts_scraped += 1
                            if contracts_scraped == max_contracts:
                                return provisions

    return provisions


def scrape_random_contracts(data_dir: str, max_contracts=10000,
                            verbose: bool = True, filtering: bool = True, stop_words: Set[str] = None) -> List[LabeledProvision]:
    """Randomly sample contracts to extract labeled provisions from"""
    if verbose:
        print('Fetching contracts from', data_dir)
    contracts = glob.glob(os.path.join(data_dir, '*/*/*/*.htm'))

    if verbose:
        print(len(contracts), 'contracts found, sampling', max_contracts)
    random.shuffle(contracts)

    contracts_scraped = 0
    provisions: List[LabeledProvision] = []

    for contract in contracts:
        if verbose:
            print('Scraping', contracts_scraped, contract)
        provisions_doc = scrape_exhibit_10(contract, filtering=filtering, stop_words=stop_words)
        if provisions_doc:
            provisions.extend(provisions_doc)
            contracts_scraped += 1
            if contracts_scraped == max_contracts:
                break

    return provisions


if __name__ == '__main__':

    data_dir = '/home/don/resources/sec_crawler/data/'
    out_dir = './'

    stop_words = set(stopwords.words('english'))

    # provisions = scrape_random_contracts(data_dir, max_contracts=max_contracts, stop_words=stop_words)
    provisions = scrape_by_year(data_dir, years=range(2019, 2015, -1), stop_words=stop_words)

    outfile = os.path.join(out_dir, 'sec_corpus_2016-2019.jsonl')
    with open(outfile, 'w', encoding='utf8') as f:
        for provision in provisions:
            json.dump(provision.__dict__, f, ensure_ascii=False)
            f.write('\n')
