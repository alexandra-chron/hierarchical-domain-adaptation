import gzip
import json
import os
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import argparse

''' Example 1 - get sorted dict of domains (per number of docs) and the subdomains for all domains:
      python preprocess_c4.py --dump_path ./ --get_number_of_docs_per_domain True --get_subdomains_for_all_domains True

    Example 2 - get sorted list of domains and the subdomains only for specific domains:
      python preprocess_c4.py --dump_path ./ --get_number_of_docs_per_domain True 
      --get_subdomains_for_specific_domains "journals.plos.org" "www.mdpi.com" "www.frontiersin.org" "www.springer.com"
    '''


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument("--data_dir", type=str, default="/raid/dirkg/c4/en/", help="Path where c4 is stored")
    parser.add_argument("--dump_path", type=str, default="./", help="Experiment dump path")
    return parser


def main(params):
    # files = json.load(open("domains_and_subdomains.json", 'r'))
    # files2 = json.load(open("sorted_domains.json", 'r'))
    files = glob(params.data_dir + 'c4-train.*')
    dict_domains = {}
    for url in ['www.frontiersin.org', 'journals.plos.org', 'www.nytimes.com', 'www.latimes.com']:
        dict_domains[url] = []
    for filename in tqdm(files, desc='tqdm() Progress Bar'):
        with gzip.open(filename, 'rb') as f:
            f = f.read().decode('utf8')
            data_dict = json.loads("[" + f.replace("}\n{", "},\n{") + "]")

            for triple in data_dict:
                url_temp = "".join(triple["url"].split("://")[1:])
                url = url_temp.split("/")[0].rstrip(".")

                if url in ['www.frontiersin.org', 'journals.plos.org', 'www.nytimes.com', 'www.latimes.com']:
                    dict_domains[url].append(triple["text"])

        json.dump(dict_domains, open(params.dump_path + "data_from_front_jou_ny_la.json", 'w'))


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    # run experiment
    main(params)
