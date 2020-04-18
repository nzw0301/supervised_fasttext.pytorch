import os
import string

import numpy as np

from supervised_fasttext.dictionary import SupervisedDictionary

PREDEFINED_VOCAB = {'a', 'b', 'c'}


def create_test_corpus_files(fname):
    """
    file content:

    label-a\ta

    label-b\tb

    label-c\tc c c

    ...

    label-f\tf ....
    """
    with open(fname, "w") as f:
        for count, char in enumerate(string.ascii_lowercase[:6], start=1):
            words = []
            for _ in range(count):
                words.append(char)

            f.write("label-{}\t{}\n".format(char, " ".join(words)))
        f.write("\n")


def delete_test_corpus_files(fname):
    os.remove(fname)


def tests():
    def tests_fit_without_ngram(fname):
        def test_fit_without_replace_and_add_special(fname):
            dictionary = SupervisedDictionary(
                replace_OOV_word=False,
                min_count=1,
                replace_word="<UNK>",
                size_word_n_gram=1,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word="</s>"
            )

            dictionary.fit(fname)

            # word vocab related test
            assert len(dictionary.word_vocab) == 7  # a b c d e f </s>
            assert dictionary.size_word_vocab == 7
            assert dictionary.num_words == np.sum(np.arange(7)) + 6

            assert dictionary.size_total_vocab == 7

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 0

            # label related test
            assert len(dictionary.label_vocab) == 6

        def test_fit_without_replace_mincount(fname):
            dictionary = SupervisedDictionary(
                replace_OOV_word=False,
                min_count=3,
                replace_word="<UNK>",
                size_word_n_gram=1,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word="</s>"
            )

            dictionary.fit(fname)

            # word vocab related test
            assert len(dictionary.word_vocab) == 5  # c d e f </s>
            assert dictionary.size_word_vocab == 5
            assert dictionary.num_words == np.sum(np.arange(7)) + 6 - 3

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 0

            # label related test
            assert len(dictionary.label_vocab) == 6

        def test_fit_with_replace_mincount(fname):
            dictionary = SupervisedDictionary(
                replace_OOV_word=True,
                min_count=3,
                replace_word="<UNK>",
                size_word_n_gram=1,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word="</s>"
            )

            dictionary.fit(fname)
            # word vocab related test
            assert len(dictionary.word_vocab) == 6  # <UNK> c d e f </s>
            assert dictionary.size_word_vocab == 6
            assert dictionary.num_words == np.sum(np.arange(7)) + 6
            assert dictionary.size_total_vocab == 6

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 0

            # label related test
            assert len(dictionary.label_vocab) == 6


        def test_fit_without_eos(fname):
            dictionary = SupervisedDictionary(
                replace_OOV_word=True,
                min_count=3,
                replace_word="<UNK>",
                size_word_n_gram=1,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)

            # word vocab related test
            assert len(dictionary.word_vocab) == 5  # <UNK> c d e f
            assert dictionary.size_word_vocab == 5
            assert dictionary.num_words == np.sum(np.arange(7))
            assert dictionary.size_total_vocab == 5

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 0

            # label related test
            assert len(dictionary.label_vocab) == 6

        def test_predefined_vocab(fname):
            # min count == 1
            dictionary = SupervisedDictionary(
                replace_OOV_word=False,
                min_count=1,
                replace_word="",
                size_word_n_gram=1,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)
            dictionary.update_vocab_from_word_set(PREDEFINED_VOCAB)

            # word vocab related test
            assert len(dictionary.word_vocab) == 3  # a b c
            assert dictionary.size_word_vocab == 3
            assert dictionary.num_words == 1 + 2 + 3
            assert dictionary.size_total_vocab == 3

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 0

            # label related test
            assert len(dictionary.label_vocab) == 6

            dictionary = SupervisedDictionary(
                replace_OOV_word=True,
                min_count=1,
                replace_word="<UNK>",
                size_word_n_gram=1,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)
            dictionary.update_vocab_from_word_set(PREDEFINED_VOCAB)

            # word vocab related test
            assert len(dictionary.word_vocab) == 4  # a b c <UNK>
            assert dictionary.size_word_vocab == 4
            assert dictionary.num_words == np.sum(np.arange(7))
            assert dictionary.size_total_vocab == 4

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 0

            # label related test
            assert len(dictionary.label_vocab) == 6

            ## min_count == 2
            dictionary = SupervisedDictionary(
                replace_OOV_word=False,
                min_count=2,
                replace_word="",
                size_word_n_gram=1,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)
            dictionary.update_vocab_from_word_set(PREDEFINED_VOCAB)

            # word vocab related test
            assert len(dictionary.word_vocab) == 2  # b c
            assert dictionary.size_word_vocab == 2
            assert dictionary.num_words == 2 + 3
            assert dictionary.size_total_vocab == 2

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 0

            # label related test
            assert len(dictionary.label_vocab) == 6

            dictionary = SupervisedDictionary(
                replace_OOV_word=True,
                min_count=2,
                replace_word="<UNK>",
                size_word_n_gram=1,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)

            print(dictionary.word_vocab.id2word)
            print(dictionary.word_vocab.word2id)
            dictionary.update_vocab_from_word_set(PREDEFINED_VOCAB)

            print(dictionary.word_vocab.id2word)
            print(dictionary.word_vocab.word2id)
            # word vocab related test
            assert len(dictionary.word_vocab) == 3  # b c <UNK>
            assert dictionary.size_word_vocab == 3
            assert dictionary.num_words == np.sum(np.arange(7))
            assert dictionary.size_total_vocab == 3

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 0

            # label related test
            assert len(dictionary.label_vocab) == 6



        test_fit_without_replace_and_add_special(fname)
        test_fit_without_replace_mincount(fname)
        test_fit_with_replace_mincount(fname)
        test_fit_without_eos(fname)
        test_predefined_vocab(fname)

    def tests_fit_with_ngram(fname):
        def test_fit_without_replacement(fname):
            dictionary = SupervisedDictionary(
                replace_OOV_word=False,
                min_count=1,
                replace_word="<UNK>",
                size_word_n_gram=2,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)

            # word vocab related test
            assert len(dictionary.word_vocab) == 6  # a b c d e f
            assert dictionary.size_word_vocab == 6
            assert dictionary.num_words == np.sum(np.arange(7))

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 5  # b-b c-c d-d e-e f-f
            assert np.sum(list(dictionary.ngram_vocab.word2freq.values())) == np.sum(np.arange(6))

            # label related test
            assert len(dictionary.label_vocab) == 6

            assert dictionary.size_total_vocab == 11

        def test_fit_without_replacement_with_mincount(fname):
            dictionary = SupervisedDictionary(
                replace_OOV_word=False,
                min_count=1,
                replace_word="<UNK>",
                size_word_n_gram=2,
                word_n_gram_min_count=2,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)

            # word vocab related test
            assert len(dictionary.word_vocab) == 6  # a b c d e f
            assert dictionary.size_word_vocab == 6
            assert dictionary.num_words == np.sum(np.arange(7))

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 4  # c-c d-d e-e f-f
            assert np.sum(dictionary.ngram_vocab.id2freq) == np.sum(np.arange(6)) - 1

            # label related test
            assert len(dictionary.label_vocab) == 6

            assert dictionary.size_total_vocab == 10

        def test_fit_without_replace_mincount(fname):
            dictionary = SupervisedDictionary(
                replace_OOV_word=False,
                min_count=3,
                replace_word="<UNK>",
                size_word_n_gram=2,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)

            # word vocab related test
            assert len(dictionary.word_vocab) == 4  # c d e f
            assert dictionary.size_word_vocab == 4
            assert dictionary.num_words == np.sum(np.arange(7)) - 3

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 4  # c-c d-d e-e f-f
            assert np.sum(dictionary.ngram_vocab.id2freq) == 2 + 3 + 4 + 5

            # label related test
            assert len(dictionary.label_vocab) == 6

            assert dictionary.size_total_vocab == 8

        def test_fit_without_replace_mincount_min_count_ngram(fname):
            dictionary = SupervisedDictionary(
                replace_OOV_word=False,
                min_count=3,
                replace_word="",
                size_word_n_gram=2,
                word_n_gram_min_count=3,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)

            # word vocab related test
            assert len(dictionary.word_vocab) == 4  # c d e f
            assert dictionary.size_word_vocab == 4
            assert dictionary.num_words == np.sum(np.arange(7)) - 3

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 3  # d-d e-e f-f
            assert np.sum(dictionary.ngram_vocab.id2freq) == 3 + 4 + 5

            # label related test
            assert len(dictionary.label_vocab) == 6

            assert dictionary.size_total_vocab == 7

        def test_fit_with_replace_mincount_min_count_ngram(fname):
            dictionary = SupervisedDictionary(
                replace_OOV_word=True,
                min_count=3,
                replace_word="<UNK>",
                size_word_n_gram=2,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)

            # word vocab related test
            assert len(dictionary.word_vocab) == 5  # c d e f <UNK>
            assert dictionary.size_word_vocab == 5
            assert dictionary.num_words == np.sum(np.arange(7))

            # n-gram related test
            assert len(dictionary.ngram_vocab) == 5  # c-c d-d e-e f-f <UNK>-<UNK>
            assert np.sum(dictionary.ngram_vocab.id2freq) == 2 + 3 + 4 + 5 + 1

            # label related test
            assert len(dictionary.label_vocab) == 6

            assert dictionary.size_total_vocab == 10

        test_fit_without_replacement(fname)
        test_fit_without_replacement_with_mincount(fname)
        test_fit_without_replace_mincount(fname)
        test_fit_without_replace_mincount_min_count_ngram(fname)
        test_fit_with_replace_mincount_min_count_ngram(fname)

    def test_transform(fname):
        def test_without_ngram(fname):
            dictionary = SupervisedDictionary(
                replace_OOV_word=False,
                min_count=2,
                replace_word="",
                size_word_n_gram=1,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)
            X, y = dictionary.transform(fname)
            assert len(X[0]) == 0
            assert y[0] == 0
            np.testing.assert_array_equal(X[-1], np.zeros(6, dtype=np.int64))
            recovered_sentence = dictionary.recover_sentence_from_ids(X[1])
            assert recovered_sentence == ["b", "b"]

        def test_with_ngram(fname):
            dictionary = SupervisedDictionary(
                replace_OOV_word=True,
                min_count=3,
                replace_word="<UNK>",
                size_word_n_gram=2,
                word_n_gram_min_count=1,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)
            X, y = dictionary.transform(fname)
            assert len(X[1]) == 3  # <unk>-<unk> <unk> <unk>
            recovered_sentence = dictionary.recover_sentence_from_ids(X[1])
            assert recovered_sentence == ["<UNK>", "<UNK>", "<UNK>-<UNK>"]

            recovered_sentence = dictionary.recover_sentence_from_ids(X[3])
            assert recovered_sentence == ["d", "d", "d", "d", "d-d", "d-d", "d-d"]


            dictionary = SupervisedDictionary(
                replace_OOV_word=True,
                min_count=3,
                replace_word="<UNK>",
                size_word_n_gram=2,
                word_n_gram_min_count=3,
                label_separator='\t',
                line_break_word=""
            )

            dictionary.fit(fname)
            X, y = dictionary.transform(fname)
            print(X)
            assert len(X[1]) == 2  # <unk> <unk>
            recovered_sentence = dictionary.recover_sentence_from_ids(X[1])
            assert recovered_sentence == ["<UNK>", "<UNK>"]

            recovered_sentence = dictionary.recover_sentence_from_ids(X[2])
            assert recovered_sentence == ["c", "c", "c"]

        test_without_ngram(fname)
        test_with_ngram(fname)


    fname = "doc.txt"
    create_test_corpus_files(fname)

    tests_fit_without_ngram(fname)
    tests_fit_with_ngram(fname)

    test_transform(fname)
    # delete_test_corpus_files(fname)
