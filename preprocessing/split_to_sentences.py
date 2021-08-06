import json


def split_to_sentences(data_path, domain, split):
    """Splits train set into train and dev sets."""
    file = json.load(open(data_path + domain + "." + split + '.json', 'r'))
    docs_list = []
    for doc in file:
        lines = doc.split('\n')
        docs_list.extend(lines)

    with open(data_path + domain + "." + split + ".sentences.txt", "w") as fobj:
        for x in docs_list:
            fobj.write(x + "\n")


path = "/home/alexandrac/projects/hierarchical-domain-adaptation/corpora/"
for split in ["val", "train", "test"]:
    for domain in ['money', 'business', 'music', 'film']:
        split_to_sentences(path, domain, split)

