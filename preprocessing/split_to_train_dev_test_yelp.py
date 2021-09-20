import json

import argparse
from numpy.random import choice


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument("--data_dir", type=str, default="/home/achron/hierarchical-domain-adaptation/"
                                                        "corpora/yelp/yelp_academic_dataset_review.json",
                        help="Path where data is stored")
    parser.add_argument("--dump_path", types=str, default="./", help="Experiment dump path")
    return parser


def main(params):
    files_new = {}
    files_new["yelp"] = []

    num_tokens_per_doc = {}
    domain = "yelp"
    ignored_docs = 0
    for line in open(params.data_dir, "r"):
        item = json.loads(line)
        num_tokens_per_doc[domain] = []
        count = 0
        doc = item["text"]
        num_tokens = len(doc.split(' '))
        if num_tokens > 200:
            count += num_tokens
            files_new[domain].append(doc)
            num_tokens_per_doc[domain].append(num_tokens)
        else:
            ignored_docs += 1

    print("{} documents were too short and thus ignored for domain {}.".format(ignored_docs, domain))

    for domain, doc_list in files_new.items():
        train_set_domain1, val_set_domain1, test_set_domain1 = [], [], []

        # create train size based on sentences length
        docs_in_domain = len(doc_list)
        train_size = int(docs_in_domain * 80 // 100)
        val_size = int(docs_in_domain * 10 // 100)
        test_size = int(docs_in_domain * 10 // 100)

        docs_to_sample = val_size + test_size

        draw = list(choice(doc_list, docs_to_sample, replace=False))

        val_set_domain1 = draw[:val_size]
        test_set_domain1 = draw[-test_size:]

        num_tokens_val = 0
        num_tokens_test = 0
        for doc in val_set_domain1:
            num_tokens_val += len(doc.split(" "))
        for doc in test_set_domain1:
            num_tokens_test += len(doc.split(" "))
        for doc in doc_list:
            if doc not in draw:
                train_set_domain1.append(doc)

        print("Train set for domain {} has {} documents.".format(domain, len(train_set_domain1)))
        print("Val set for domain {} has {} documents.".format(domain, len(val_set_domain1)))
        print("Test set for domain {} has {} documents.".format(domain, len(test_set_domain1)))

        print("The dev set has {} and the test set has {} tokens.".format(num_tokens_val, num_tokens_test))

        # print("yo")
        domain_temp = domain.split("/")[-1]
        json.dump(train_set_domain1,
                  open(params.dump_path + "{}.train.json".format(domain_temp), 'w'))
        json.dump(val_set_domain1,
                  open(params.dump_path + "{}.val.json".format(domain_temp), 'w'))
        json.dump(test_set_domain1,
                  open(params.dump_path + "{}.test.json".format(domain_temp), 'w'))


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    # run experiment
    main(params)
