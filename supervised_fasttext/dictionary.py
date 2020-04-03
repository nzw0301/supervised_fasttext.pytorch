import logging

import numpy as np
from tokenizer.vocab import Vocab


class SupervisedDictionary(object):

    def __init__(
            self,
            replace_lower_freq_word=False,
            min_count=5,
            replace_word="<UNK>",
            size_word_n_gram=1,
            word_n_gram_min_count=1,
            label_separator='\t',
            line_break_word="</s>"
    ):
        """
        :param replace_lower_freq_word: boolean. Whether replace lower frequency and OOV word with `replace_word`.
            If False, these words removed from sequences.
        :param min_count: Threshold of word frequency.
        :param replace_word: str. Replacing word for OOV word.
        :param size_word_n_gram:
        :param word_n_gram_min_count:
        """
        self.word_vocab = Vocab(replace_lower_freq_word, replace_word)
        self.ngram_vocab = Vocab(False, None)
        self.label_vocab = Vocab(False, None)
        self.num_words = 0
        self.num_vocab = 0
        self.replace_lower_freq_word = replace_lower_freq_word
        self.min_count = max(min_count, 1)
        self.replace_word = replace_word
        self.is_tokenized = False
        self.size_word_n_gram = max(size_word_n_gram, 1)
        self.word_n_gram_min_count = max(word_n_gram_min_count, 1)
        self.label_separator = label_separator
        self.line_break_word = line_break_word

        # add special words to vocab
        if self.replace_lower_freq_word and self.replace_word:
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
        :param docs: List of str
        :return: None
        """
        if self.is_tokenized:
            self.logger.warning("Warning: this instance has already fitted.")

        with open(fname) as f:
            for doc in f:
                elements = doc.split(self.label_separator)
                if len(elements) == 1:
                    continue
                sequence, label = elements

                sequence += " " + self.line_break_word
                for word in sequence.split():
                    self.word_vocab.add_word(word=word)
                self.label_vocab.add_word(word=label)

            self._update_words()

            if self.size_word_n_gram > 1:
                f.seek(0)
                for doc in f:
                    sentence = doc.split(self.label_separator)[0] + " " + self.line_break_word
                    processed_words = self._sentence2cleaned_words(sentence)
                    for t in range(len(processed_words) - self.size_word_n_gram + 1):
                        for n in range(2, self.size_word_n_gram + 1):
                            self.ngram_vocab.add_word('-'.join(processed_words[t:t + n]))
                self.ngram_vocab.remove_low_freq_words(self.word_n_gram_min_count)

    def _sentence2cleaned_words(self, words: list):
        processed_words = []
        for word in words.split():
            if word in self.word_vocab.word2id:
                processed_words.append(word)
            elif self.replace_lower_freq_word:
                processed_words.append(self.replace_word)
        return processed_words

    def _words2word_ids(self, words: list):
        return [self.word_vocab.word2id[word] for word in words]

    def _words2ngram_ids(self, words: str):
        ngram_ids = []
        for t in range(len(words) - self.size_word_n_gram + 1):
            for n in range(2, self.size_word_n_gram + 1):
                ngram = '-'.join(words[t:t + n])
                if ngram in self.ngram_vocab.word2id:
                    ngram_ids.append(self.ngram_vocab.word2id[ngram])
        return ngram_ids

    def transform(self, fname: str):
        """
        :param fname:
        :return:
        """
        if not self.is_tokenized:
            raise Exception("This dictionary instance has not tokenized yet.")

        X = []
        y = []
        with open(fname) as f:
            for doc in f:
                elements = doc.split(self.label_separator)
                if len(elements) == 1:
                    continue
                sentence, label = elements
                sentence = sentence + " " + self.line_break_word
                words = self._sentence2cleaned_words(sentence)
                word_ids = np.array(self._words2word_ids(words), dtype=np.int64)
                ngram_ids = np.array(self._words2ngram_ids(words), dtype=np.int64) + self.num_vocab
                X.append(np.hstack((word_ids, ngram_ids)))
                y.append(self.label_vocab.word2id[label])

        return X, y

    def _update_words(self):
        """
        Update word related attributes.
        :return: None
        """
        self.word_vocab.remove_low_freq_words(min_count=self.min_count)
        self.num_vocab = len(self.word_vocab)
        self.num_words = np.sum(self.word_vocab.id2freq)
        self.is_tokenized = True

    def recover_sentence_from_ids(self, word_ids: np.ndarray):
        words = []
        for w_id in word_ids:
            if w_id >= self.num_vocab:
                words.append(self.ngram_vocab.id2word[w_id-self.num_vocab])
            else:
                words.append(self.word_vocab.id2word[w_id])
        return words

