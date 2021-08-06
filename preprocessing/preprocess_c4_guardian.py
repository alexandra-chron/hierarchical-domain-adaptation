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
    # parser.add_argument("--get_number_of_docs_per_domain", type=bool, default=False,
    #                     help="Create a sorted json that has every url and the number of corresponding docs"
    #                          "from most to least frequent")
    # parser.add_argument("--get_subdomains_for_all_domains", type=bool, default=False,
    #                     help="Gather subdomains (after `/` in the url) and their content (text)")
    # parser.add_argument("--get_subdomains_for_specific_domains", nargs="+",
    #                     default=None,
    #                     help="Get the subdomains (after `/` in the url) for the domains specified here (in a list)")
    return parser


def main(params):
    # files = json.load(open("domains_and_subdomains.json", 'r'))
    # files2 = json.load(open("sorted_domains.json", 'r'))
    files = glob(params.data_dir + 'c4-train.*')
    dict_domains = {}
    count = 0
    for filename in tqdm(files, desc='tqdm() Progress Bar'):
        with gzip.open(filename, 'rb') as f:
            f = f.read().decode('utf8')
            data_dict = json.loads("[" + f.replace("}\n{", "},\n{") + "]")

            for triple in data_dict:
                url_temp = "".join(triple["url"].split("://")[1:])
                url = url_temp.split("/")[0].rstrip(".")
                if len(url_temp.split("/")) >= 2:
                    url_2 = url_temp.split("/")[1].rstrip(".")

                    if url == 'www.theguardian.com':
                        if url_2 not in dict_domains.keys():
                            dict_domains[url_2] = []
                            dict_domains[url_2].append(triple["text"])
                        else:
                            dict_domains[url_2].append(triple["text"])
                        count += 1
        json.dump(dict_domains, open(params.dump_path + "domains_and_subdomains_guardian.json", 'w'))


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    # run experiment
    main(params)
