import gzip
import json
import os
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import argparse

''' Example 1 - get sorted dict of domains (per number of docs) and the subdomains for all domains:
      python preprocess_c4_springer.py --dump_path ./ --get_number_of_docs_per_domain True --get_subdomains_for_all_domains True

    Example 2 - get sorted list of domains and the subdomains only for specific domains:
      python preprocess_c4_springer.py --dump_path ./ --get_number_of_docs_per_domain True 
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
    # domain_with_subdomains = {'www.frontiersin.org': ['journals/sociology', 'journals/psychology'],
    #                           'journals.plos.org': ['plosbiology', 'plosmedicine']}
    dict_domains = {}
    count = 0
    dict_domains['www.springer.com/psychology'] = []
    dict_domains['www.springer.com/philosophy'] = []
    dict_domains['www.springer.com/social+sciences'] = []
    dict_domains['www.springer.com/mathematics'] = []
    dict_domains['www.springer.com/physics'] = []
    dict_domains['www.springer.com/economics'] = []

    for filename in tqdm(files, desc='tqdm() Progress Bar'):
        with gzip.open(filename, 'rb') as f:
            f = f.read().decode('utf8')
            data_dict = json.loads("[" + f.replace("}\n{", "},\n{") + "]")

            for triple in data_dict:
                url_temp = "".join(triple["url"].split("://")[1:])
                url = url_temp.split("/")[0].rstrip(".")
                if len(url_temp.split("/")) >= 2:
                    url_2 = url_temp.split("/")[1].rstrip(".")

                    if url == 'www.springer.com':
                        # print(url_2)
                        if url_2 == 'psychology':
                            dict_domains['www.springer.com/psychology'].append(triple["text"])

                        elif url_2 == 'philosophy':
                            dict_domains['www.springer.com/philosophy'].append(triple["text"])

                        elif url_2 == 'social+sciences':
                            dict_domains['www.springer.com/social+sciences'].append(triple["text"])
                        #
                        elif url_2 == 'mathematics':
                            dict_domains['www.springer.com/mathematics'].append(triple["text"])

                        elif url_2 == 'physics':
                            dict_domains['www.springer.com/physics'].append(triple["text"])

                        elif url_2 == 'economics':
                            dict_domains['www.springer.com/economics'].append(triple["text"])
        count += 1
        # if count % 10 == 0 and count <= 50:
            # json.dump(dict_domains, open(params.dump_path + "domains_and_subdomains_{}_files.json".format(count), 'w'))

        json.dump(dict_domains, open(params.dump_path + "domains_and_subdomains_springer.json", 'w'))


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    # run experiment
    main(params)
