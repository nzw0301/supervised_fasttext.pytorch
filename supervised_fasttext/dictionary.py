import logging

import numpy as np
from tokenizer.vocab import Vocab


class SupervisedDictionary(object):

    def __init__(
            self,
            replace_OOV_word=False,
            min_count=5,
            replace_word="<UNK>",
            size_word_n_gram=1,
            word_n_gram_min_count=1,
            label_separator='\t',
            line_break_word="</s>",
    ):
        """
        :param replace_OOV_word: boolean. Whether replace OOV words with `replace_word`.
            If False, these words removed from sequences.
        :param min_count: Threshold of word frequency.
        :param replace_word: str. Replacing word for OOV word.
        :param size_word_n_gram: the maximum ngram length.
        :param word_n_gram_min_count: Threshold of n-gram frequency.
        :param label_separator: str. Separator between label and sentence.
        :param line_break_word: special token added into the end of sentence.
        """
        self.word_vocab = Vocab(replace_OOV_word, replace_word)
        self.ngram_vocab = Vocab(False, None)
        self.label_vocab = Vocab(False, None)
        self.num_words = 0
        self.size_word_vocab = 0
        self.size_ngram_vocab = 0
        self.size_total_vocab = 0
        self.replace_OOV_word = replace_OOV_word
        self.min_count = max(min_count, 1)
        self.replace_word = replace_word
        self.is_tokenized = False
        self.size_word_n_gram = max(size_word_n_gram, 1)
        self.word_n_gram_min_count = max(word_n_gram_min_count, 1)
        self.label_separator = label_separator
        self.line_break_word = line_break_word

        # add special words to vocab
        if self.replace_OOV_word and self.replace_word:
            self.word_vocab.id2word.append(self.replace_word)
            self.word_vocab.word2id[self.replace_word] = len(self.word_vocab.word2id)

        # init logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        self.logger = logger

    def fit(self, fname: str):
        """
        Fit on list of str.
        :param fname: str. File name. In the file, each line must be `label[SEP]Sentence`.
        :return: None
        """
        if self.is_tokenized:
            self.logger.warning("Warning: this instance has already fitted.")

        with open(fname) as f:
            for line_id, line in enumerate(f, start=1):
                elements = line.strip().split(sep=self.label_separator, maxsplit=2)
                if len(elements) == 1:
                    self.logger.warning('{}th line is empty'.format(line_id))
                    continue
                label, sentence = elements
                sentence += " " + self.line_break_word
                for word in sentence.split():
                    self.word_vocab.add_word(word=word)
                self.label_vocab.add_word(word=label)

            self._update_words()

            if self.size_word_n_gram > 1:
                f.seek(0)
                for line_id, line in enumerate(f):
                    elements = line.strip().split(sep=self.label_separator, maxsplit=2)
                    if len(elements) == 1:
                        self.logger.warning('{}th line is empty'.format(line_id))
                        continue
                    sentence = elements[1] + " " + self.line_break_word
                    processed_words = self._sentence2cleaned_words(sentence)
                    for t in range(len(processed_words) - self.size_word_n_gram + 1):
                        for n in range(2, self.size_word_n_gram + 1):
                            self.ngram_vocab.add_word('-'.join(processed_words[t:t + n]))
                self.ngram_vocab.remove_low_freq_words(self.word_n_gram_min_count)
                self.size_ngram_vocab = len(self.ngram_vocab)
        self.size_total_vocab = self.size_word_vocab + self.size_ngram_vocab

    def _sentence2cleaned_words(self, sentence: str):
        """
        Convert str into list of str. Words in list do not contain OOV words.
        :param words: Str. sentence sample.
        :return: List of str.
        """
        processed_words = []
        for word in sentence.split():
            if word in self.word_vocab.word2id:
                processed_words.append(word)
            elif self.replace_OOV_word:
                processed_words.append(self.replace_word)
        return processed_words

    def _words2word_ids(self, words: list):
        """
        :param words: list of cleaned words, which is list of str.
        :return: list of word ids.
        """
        return [self.word_vocab.word2id[word] for word in words]

    def _words2ngram_ids(self, words: str):
        """
        :param words: list of cleaned words, which is list of str.
        :return: list of n-gram ids.
        """
        ngram_ids = []
        for t in range(len(words) - self.size_word_n_gram + 1):
            for n in range(2, self.size_word_n_gram + 1):
                ngram = '-'.join(words[t:t + n])
                if ngram in self.ngram_vocab.word2id:
                    ngram_ids.append(self.ngram_vocab.word2id[ngram] + self.size_word_vocab)
        return ngram_ids

    def transform(self, fname: str):
        """
        :param fname: str. File name.
        :return: tuple of two lists. Lists are sentences and labels respectively.
        """
        if not self.is_tokenized:
            raise Exception("This dictionary instance has not tokenized yet.")

        X = []
        y = []
        with open(fname) as f:
            for line in f:
                elements = line.strip().split(sep=self.label_separator, maxsplit=2)
                if len(elements) == 1:
                    label, sentence = elements[0], ""
                else:
                    label, sentence = elements
                sentence = sentence + " " + self.line_break_word
                words = self._sentence2cleaned_words(sentence)
                word_ids = np.array(self._words2word_ids(words), dtype=np.int64)
                ngram_ids = np.array(self._words2ngram_ids(words), dtype=np.int64)
                X.append(np.hstack((word_ids, ngram_ids)))
                y.append(self.label_vocab.word2id[label])

        return X, y

    def _update_words(self):
        """
        Update word related attributes.
        :return: None
        """
        self.word_vocab.remove_low_freq_words(min_count=self.min_count)
        self.size_word_vocab = len(self.word_vocab)
        self.num_words = np.sum(self.word_vocab.id2freq)
        self.is_tokenized = True

    def recover_sentence_from_ids(self, word_ids: np.ndarray):
        words = []
        for w_id in word_ids:
            if w_id >= self.size_word_vocab:
                words.append(self.ngram_vocab.id2word[w_id - self.size_word_vocab])
            else:
                words.append(self.word_vocab.id2word[w_id])
        return words

    def update_vocab_from_word_set(self, predefined_set: set) -> None:
        """
        Remove or replace word by using pre-defined word vocab such as pre-trained model's vocab.s
        :param predefined_set: Set of str.
        :return: None
        """
        assert self.is_tokenized

        if self.replace_OOV_word:
            new_id2word = []
            new_id2freq = []
            total_freq_replaced_words = 0
            for word in self.word_vocab.id2word:
                if word in predefined_set:
                    new_id2word.append(word)
                    new_id2freq.append(self.word_vocab.word2freq[word])
                else:
                    total_freq_replaced_words += self.word_vocab.word2freq[word]

            # OOV is already skipped, so it is added again here
            new_id2word.append(self.replace_word)
            new_id2freq.append(total_freq_replaced_words)

            self.word_vocab.id2word = new_id2word
            self.word_vocab.id2freq = new_id2freq
            self.word_vocab.word2id = {word: word_id for word_id, word in enumerate(self.word_vocab.id2word)}
            self.word2freq = {
                self.word_vocab.id2word[word_id]: freq for word_id, freq in enumerate(self.word_vocab.id2freq)
            }

        else:
            for word in self.word_vocab.id2word:
                if word not in predefined_set:
                    self.word_vocab.word2freq[word] = 0
            self._update_words()

        self.size_word_vocab = len(self.word_vocab)
        self.num_words = np.sum(self.word_vocab.id2freq)
        self.size_total_vocab = self.size_word_vocab + self.size_ngram_vocab
