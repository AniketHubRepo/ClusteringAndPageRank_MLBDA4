# M25DE1051 | Aniket Srivastava | Assignment 4 | Part 2: Web Search
# CSL7110: Machine Learning with Big Data

import os
import math
import re
import string

# Stop words and punctuation as defined in assignment
STOP_WORDS = {
    'a', 'an', 'the', 'they', 'these', 'this', 'for', 'is', 'are',
    'was', 'of', 'or', 'and', 'does', 'will', 'whose'
}

PUNCTUATION = set('{}[]<>=(). ,;\'"?#!-:')

# Plural/singular normalization map (exhaustive as per assignment)
PLURAL_MAP = {
    'stacks': 'stack', 'structures': 'structure', 'applications': 'application',
    'elements': 'element', 'operations': 'operation', 'implementations': 'implementation',
    'items': 'item', 'collections': 'collection', 'functions': 'function',
    'pages': 'page', 'words': 'word', 'indices': 'index', 'entries': 'entry',
    'engineers': 'engineer', 'plates': 'plate', 'terms': 'term',
    'webpages': 'webpage', 'queries': 'query', 'lists': 'list',
    'pushes': 'push', 'pops': 'pop', 'magazines': 'magazine',
}


def normalize(word):
    """Lowercase and apply singular normalization."""
    w = word.lower()
    return PLURAL_MAP.get(w, w)


def tokenize(text):
    """
    Tokenize raw text into (position, normalized_word) pairs.
    Replaces punctuation with spaces, filters stop words.
    Returns list of (index, word) keeping all positions for index tracking.
    """
    for ch in PUNCTUATION:
        text = text.replace(ch, ' ')
    raw_tokens = text.split()
    result = []
    pos = 0
    for raw in raw_tokens:
        pos += 1
        w = normalize(raw)
        if w and w not in STOP_WORDS:
            result.append((pos, w))
    return result, pos  # (token list, total word count including stops)


# Position: stores page reference and word index
class Position:
    def __init__(self, page_entry, word_index):
        self._page = page_entry
        self._word_index = word_index

    def getPageEntry(self):
        return self._page

    def getWordIndex(self):
        return self._word_index

    def __repr__(self):
        return f"({self._page.pageName}, {self._word_index})"


# WordEntry: stores all positions for a given word
class WordEntry:
    def __init__(self, word):
        self.word = word
        self._positions = []

    def addPosition(self, position):
        self._positions.append(position)

    def addPositions(self, positions):
        self._positions.extend(positions)

    def getAllPositionsForThisWord(self):
        return list(self._positions)

    def getTermFrequency(self, page_entry):
        """TF = occurrences of word in page / total words in page."""
        count = sum(1 for p in self._positions if p.getPageEntry() is page_entry)
        total = page_entry.totalWords
        return count / total if total > 0 else 0.0

    def __repr__(self):
        return f"WordEntry({self.word}, positions={len(self._positions)})"


# MySet: set with union and intersection
class MySet:
    def __init__(self):
        self._items = []

    def addElement(self, element):
        if element not in self._items:
            self._items.append(element)

    def union(self, other_set):
        result = MySet()
        for item in self._items:
            result.addElement(item)
        for item in other_set._items:
            result.addElement(item)
        return result

    def intersection(self, other_set):
        result = MySet()
        other_items = other_set._items
        for item in self._items:
            if item in other_items:
                result.addElement(item)
        return result

    def toList(self):
        return list(self._items)

    def __len__(self):
        return len(self._items)


# PageIndex: inverted index for a single page
class PageIndex:
    def __init__(self):
        self._word_entries = {}

    def addPositionForWord(self, word, position):
        if word not in self._word_entries:
            self._word_entries[word] = WordEntry(word)
        self._word_entries[word].addPosition(position)

    def getWordEntries(self):
        return list(self._word_entries.values())

    def getWordEntry(self, word):
        return self._word_entries.get(word, None)

    def containsWord(self, word):
        return word in self._word_entries

    def getPositions(self, word):
        we = self._word_entries.get(word)
        return we.getAllPositionsForThisWord() if we else []


# MyHashTable: maps word -> WordEntry across all pages
class MyHashTable:
    def __init__(self, size=1024):
        self._size = size
        self._table = [None] * size

    def getHashIndex(self, word):
        h = 0
        for ch in word:
            h = (h * 31 + ord(ch)) % self._size
        return h

    def addPositionsForWord(self, word_entry):
        idx = self.getHashIndex(word_entry.word)
        if self._table[idx] is None:
            self._table[idx] = {}
        if word_entry.word in self._table[idx]:
            self._table[idx][word_entry.word].addPositions(
                word_entry.getAllPositionsForThisWord()
            )
        else:
            self._table[idx][word_entry.word] = word_entry

    def getWordEntry(self, word):
        idx = self.getHashIndex(word)
        if self._table[idx] is None:
            return None
        return self._table[idx].get(word, None)

    def getAllWords(self):
        words = []
        for bucket in self._table:
            if bucket:
                words.extend(bucket.keys())
        return words


# PageEntry: reads a webpage file, builds its page index
class PageEntry:
    def __init__(self, page_name, webpages_dir):
        self.pageName = page_name
        filepath = os.path.join(webpages_dir, page_name)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        tokens, self.totalWords = tokenize(text)
        self._page_index = PageIndex()
        for pos, word in tokens:
            position = Position(self, pos)
            self._page_index.addPositionForWord(word, position)

    def getPageIndex(self):
        return self._page_index


# InvertedPageIndex: global index across all added pages
class InvertedPageIndex:
    def __init__(self):
        self._pages = {}       # pageName -> PageEntry
        self._hash_table = MyHashTable()
        self._total_pages = 0

    def addPage(self, page_entry):
        self._pages[page_entry.pageName] = page_entry
        self._total_pages += 1
        for word_entry in page_entry.getPageIndex().getWordEntries():
            self._hash_table.addPositionsForWord(word_entry)

    def getPagesWhichContainWord(self, word):
        nw = normalize(word)
        result = MySet()
        for page in self._pages.values():
            if page.getPageIndex().containsWord(nw):
                result.addElement(page)
        return result

    def getTotalPages(self):
        return self._total_pages

    def getPage(self, page_name):
        return self._pages.get(page_name, None)

    def getAllPages(self):
        return list(self._pages.values())


# SearchEngine: main interface for actions
class SearchEngine:
    def __init__(self, webpages_dir):
        self._index = InvertedPageIndex()
        self._webpages_dir = webpages_dir

    def performAction(self, action_message):
        """Parse and execute an action string, return output string."""
        parts = action_message.strip().split()
        if not parts:
            return ""

        action = parts[0]

        # Action: addPage x
        if action == "addPage" and len(parts) >= 2:
            page_name = parts[1]
            page_entry = PageEntry(page_name, self._webpages_dir)
            self._index.addPage(page_entry)
            return f"Added page: {page_name}"

        # Action: queryFindPagesWhichContainWord x
        elif action == "queryFindPagesWhichContainWord" and len(parts) >= 2:
            word = parts[1]
            nw = normalize(word)
            pages = self._index.getPagesWhichContainWord(nw)
            page_list = pages.toList()
            if not page_list:
                return f"No webpage contains word {word}"
            names = sorted([p.pageName for p in page_list])
            return ", ".join(names)

        # Action: queryFindPositionsOfWordInAPage x y
        elif action == "queryFindPositionsOfWordInAPage" and len(parts) >= 3:
            word = parts[1]
            page_name = parts[2]
            nw = normalize(word)
            page = self._index.getPage(page_name)
            if page is None:
                return f"No webpage {page_name} found"
            positions = page.getPageIndex().getPositions(nw)
            if not positions:
                return f"Webpage {page_name} does not contain word {word}"
            indices = sorted([p.getWordIndex() for p in positions])
            return ", ".join(map(str, indices))

        return f"Unknown action: {action}"

    def computeTFIDF(self, word, page_name):
        """Compute TFIDF score for a word in a given page."""
        nw = normalize(word)
        page = self._index.getPage(page_name)
        if page is None:
            return 0.0
        page_idx = page.getPageIndex()
        if not page_idx.containsWord(nw):
            return 0.0
        we_global = self._index._hash_table.getWordEntry(nw)
        if we_global is None:
            return 0.0
        # TF: occurrences in page / total words in page
        positions_in_page = page_idx.getPositions(nw)
        tf = len(positions_in_page) / page.totalWords if page.totalWords > 0 else 0
        # IDF: log(N / nw)
        all_pages = self._index.getAllPages()
        n_containing = sum(1 for p in all_pages if p.getPageIndex().containsWord(nw))
        N = self._index.getTotalPages()
        idf = math.log(N / n_containing) if n_containing > 0 else 0.0
        return tf * idf


# Main driver: process actions.txt and compare with answers.txt
def main():
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Q2'))
    WEBPAGES_DIR = os.path.join(BASE, 'webpages')
    ACTIONS_FILE = os.path.join(BASE, 'actions.txt')
    ANSWERS_FILE = os.path.join(BASE, 'answers.txt')

    print("=" * 60)
    print("CSL7110 Assignment 4  |  Part 2: Web Search (Inverted Index)")
    print("=" * 60)

    engine = SearchEngine(WEBPAGES_DIR)

    with open(ACTIONS_FILE, 'r') as f:
        actions = [line.strip() for line in f if line.strip()]
    with open(ANSWERS_FILE, 'r') as f:
        answers = [line.strip() for line in f if line.strip()]

    query_outputs = []
    print("\nProcessing actions:\n")
    for action in actions:
        result = engine.performAction(action)
        print(f"  > {action}")
        if not action.startswith("addPage"):
            print(f"    Output : {result}")
            query_outputs.append(result)

    print("\n" + "=" * 60)
    print("Validation Against answers.txt")
    print("=" * 60)
    all_correct = True
    for i, (out, exp) in enumerate(zip(query_outputs, answers)):
        match = out.strip() == exp.strip()
        status = "PASS" if match else "FAIL"
        if not match:
            all_correct = False
        print(f"  Q{i+1}: [{status}]")
        if not match:
            print(f"    Expected : {exp}")
            print(f"    Got      : {out}")

    print()
    if all_correct:
        print("  All outputs match answers.txt.")
    else:
        print("  Some outputs differ from answers.txt.")
    print()


if __name__ == "__main__":
    main()
