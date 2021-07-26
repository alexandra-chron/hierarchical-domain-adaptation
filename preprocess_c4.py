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
      --get_subdomains_for_specific_domains ["journals.plos.org", "www.mdpi.com", "www.frontiersin.org", "www.springer.com"]
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
    parser.add_argument("--get_number_of_docs_per_domain", type=bool, default=False,
                        help="Create a sorted json that has every url and the number of corresponding docs"
                             "from most to least frequent")
    parser.add_argument("--get_subdomains_for_all_domains", type=bool, default=False,
                        help="Gather subdomains (after `/` in the url) and their content (text)")
    parser.add_argument("--get_subdomains_for_specific_domains", nargs="+",
                        default=None,
                        help="Get the subdomains (after `/` in the url) for the domains specified here (in a list)")
    return parser


def main(params):

    docs_per_domain = {}
    files = glob(params.data_dir + 'c4-train.*')
    dict_domains = {}

    for filename in tqdm(files, desc='tqdm() Progress Bar'):
        with gzip.open(filename, 'rb') as f:
            f = f.read().decode('utf8')
            data_dict = json.loads("[" + f.replace("}\n{", "},\n{") + "]")

            for triple in data_dict:
                url_temp = "".join(triple["url"].split("://")[1:])
                url = url_temp.split("/")[0].rstrip(".")

                if params.get_number_of_docs_per_domain:
                    if url in docs_per_domain.keys():
                        docs_per_domain[url] = docs_per_domain[url] + 1
                    else:
                        docs_per_domain[url] = 1

                if params.get_subdomains_for_all_domains or params.get_subdomains_for_specific_domains:
                    if len(url_temp.split("/")) > 1:
                        subdomain = url_temp.split("/")[1]
                    if len(url_temp.split("/")) > 2:
                        subsubdomain = url_temp.split("/")[2]
                    if len(url_temp.split("/")) > 3:
                        subsubsubdomain = url_temp.split("/")[3]

                    if (not params.get_subdomains_for_all_domains and url in params.get_subdomains_for_specific_domains) \
                            or params.get_subdomains_for_all_domains:

                            if url not in dict_domains.keys():
                                dict_domains[url] = {}
                                if len(url_temp.split("/")) > 1:
                                    # I get rid of decimals because if a link looks like this
                                    # 'www.mdpi.com/2218/9/3/52', I do not care about the subdomains (not useful)
                                    if not subdomain.isdecimal():
                                        if subdomain in dict_domains[url].keys():
                                            dict_domains[url][subdomain]["text"].extend(triple["text"])
                                        else:
                                            dict_domains[url][subdomain] = {}
                                            dict_domains[url][subdomain]["sub"] = {}
                                            dict_domains[url][subdomain]["text"] = [triple["text"]]
                                    else:
                                        continue
                                if len(url_temp.split("/")) > 2:
                                    if not subsubdomain.isdecimal():
                                        if subsubdomain in dict_domains[url][subdomain]["sub"].keys():
                                            dict_domains[url][subdomain]["sub"][subsubdomain]["text"].extend(triple["text"])
                                        else:
                                            dict_domains[url][subdomain]["sub"][subsubdomain] = {}
                                            dict_domains[url][subdomain]["sub"][subsubdomain]["sub"] = {}
                                            dict_domains[url][subdomain]["sub"][subsubdomain]["text"] = [triple["text"]]
                                    else:
                                        continue
                                if len(url_temp.split("/")) > 3:
                                    if not subsubsubdomain.isdecimal():
                                        if subsubsubdomain in dict_domains[url][subdomain]["sub"][subsubdomain]["sub"].keys():
                                            dict_domains[url][subdomain]["sub"][subsubdomain]["sub"][subsubsubdomain]["text"].extend(triple["text"])
                                        else:
                                            dict_domains[url][subdomain]["sub"][subsubdomain]["sub"][subsubsubdomain] = {}
                                            dict_domains[url][subdomain]["sub"][subsubdomain]["sub"][subsubsubdomain]["sub"] = {}
                                            dict_domains[url][subdomain]["sub"][subsubdomain]["sub"][subsubsubdomain]["text"] = [triple["text"]]
                                    else:
                                        continue

    if params.get_subdomains_for_all_domains or params.get_subdomains_for_specific_domains:
        json.dump(dict_domains, open(params.dump_path + "domains_and_subdomains.json", 'w'))
    if params.get_number_of_docs_per_domain:
        sorted_domains = sorted(docs_per_domain.items(), key=lambda item: item[1], reverse=True)
        json.dump(sorted_domains, open(params.dump_path + "sorted_domains.json", 'w'))


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    # run experiment
    main(params)
