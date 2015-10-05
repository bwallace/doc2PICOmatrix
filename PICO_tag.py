import os 
import pdb

import pandas

import PICO_robot

from PICO_robot import PICORobot

robot = PICORobot()

def tag_full_text(fpath): 
    pass 


def tag_cohen(path="input_data/cohen/ACEInhibitors_processed.csv", out_dir="tagged_data/cohen/"):
    citation_data = pandas.read_csv(path, header=None, 
                        names =["pmid", "title", "authors", "journal", "abstract", "keywords", "label"])
    

    #citation_data["title_and_abstract"] = 
    #citation_data["tags"]

    def _get_sentences(tagged_sentences):
        return [s['content'] for s in tagged_sentences]

    
    p_list, i_list, o_list = [], [], []
    pmids, labels = [], []

    for pmid, title, abstract, label in citation_data[["pmid", "title", "abstract", "label"]].values:
        cur_annotations = robot.annotate(". ".join((title, abstract)), min_k=1)['marginalia']

        p, i, o = [a['annotations'] for a in cur_annotations]

        p_sentences = _get_sentences(p)
        i_sentences = _get_sentences(i)
        o_sentences = _get_sentences(o)

        p_list.append(" ".join(p_sentences))
        i_list.append(" ".join(i_sentences))
        o_list.append(" ".join(o_sentences))

        labels.append(label)
        pmids.append(pmid)

    df = pandas.DataFrame({"pmid":pmids, "P":p_list, "I":i_list, "O":o_list, "y":labels})
    df.to_csv(os.path.join(out_dir, 
                "%s_PMID.tsv" % os.path.split(path)[-1].replace(".csv","")), sep="\t")


