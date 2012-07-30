### Copyright 2011, 2012 Margaret Mitchell
### Distributed under the terms of the GNU General Public License
###
### This file is part of the vision-to-language system Midge.
### 
### Midge is free software: you can redistribute it and/or modify
### it under the terms of the GNU General Public License as published by
### the Free Software Foundation, either version 3 of the License, or
### (at your option) any later version.
### 
### Midge is distributed in the hope that it will be useful,
### but WITHOUT ANY WARRANTY; without even the implied warranty of
### MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
### GNU General Public License for more details.
### 
### You should have received a copy of the GNU General Public License
### along with Midge.  If not, see <http://www.gnu.org/licenses/>.
### 
### Please cite the relevant work:                    
### Mitchell et al. (2012).  "Midge: Generating Image Descriptions From Computer Vision Detections." Proceedings of EACL 2012.
###
### Questions/Comments, send to m.mitchell@abdn.ac.uk

import sys
import re
import glob
import math
import itertools
from nltk.corpus import wordnet as wn
import pickle

class queryKB():
    def __init__(self, reserved_words=[], word_thresh=.001, cnt=10, read_pickle=True, read_db=False, db=None):
        self.read_pickle = read_pickle
        #print "read_pickle is", read_pickle
        self.word_thresh = float(word_thresh)
        self.cnt = cnt
        # Only generate contentful verbs for now.
        self.light_verbs = ("may", "might", "must", "would", "be", "being", "been", "am", "are", "were", "do", "does", "did", "should", "could", "would", "may",  "is", "'s", "was", "has", "had", "will", "can", "shall")
        # So we only read in info about nouns we've run detectors for.
        self.reserved_words = reserved_words
        # Maps detection names to noun names.
        self.label_hash = {'motorbike':'motorcycle', 'television':'tv', 'pottedplant':'plant'}
        self.present_tense = ("VBG", "VBZ", "VBN")
        self.visual_thresh_hash = {}
        self.hypernym_hash = {}
        self.obj_probs = {}
        self.mod_ngram_hash = {5:{}, 4:{},3:{},2:{},1:{}}
        self.plural_hash = {}
        self.mod_hash = {}
        self.det_hash = {}
        self.att_hash = {}
        self.preps = {}
        self.verb_hash = {'a':{}, 'b':{}}
        self.prep_hash = {'verb-prep':{}, 'prep-noun':{}, 'noun-prep':{}}
        self.verb_trans_hash = {}
        self.noun_freq_hash = {}
        self.noun_noun_hash = {}
        self.ins_hash = {}
        # -- Text files we learn from -- #
        self.visual_thresh_file = None
        self.mod_ngram_file = None
        self.prep_file = None
        self.det_file = None
        self.mod_file = None
        self.att_files = None
        self.verb_noun_file = None
        self.noun_verb_file = None
        self.verb_prep_file = None
        self.prep_noun_file = None
        self.noun_prep_file = None
        self.verb_trans_file = None
        self.nouns_file = None
        self.plurals_file = None
        self.ins_file = None
        if read_db:
            self.read_as_db(db)
        elif self.read_pickle:
            self.read_as_pickle()
        else:
            self.read_raw_data()

    def read_as_pickle(self):
        print "loading models..."
        print "1"
        self.visual_thresh_hash = pickle.load(open("pickled_files/visual_thresh_hash.pk", "rb"))
        print "2"
        self.hypernym_hash = pickle.load(open("pickled_files/hypernym_hash.pk", "rb"))
        print "3"
        self.obj_probs = pickle.load(open("pickled_files/obj_probs.pk", "rb"))
        print "4"
        self.mod_ngram_hash = pickle.load(open("pickled_files/mod_ngram_hash.pk", "rb"))
        print "5"
        self.plural_hash = pickle.load(open("pickled_files/plural_hash.pk", "rb"))
        print "6"
        self.mod_hash = pickle.load(open("pickled_files/mod_hash.pk", "rb"))
        print "7"
        self.det_hash = pickle.load(open("pickled_files/det_hash.pk", "rb"))
        print "8"
        self.att_hash = pickle.load(open("pickled_files/att_hash.pk", "rb"))
        print "9"
        self.preps = pickle.load(open("pickled_files/preps.pk", "rb"))
        print "10"
        self.verb_hash = pickle.load(open("pickled_files/verb_hash.pk", "rb"))
        print "11"
        self.prep_hash = pickle.load(open("pickled_files/prep_hash.pk", "rb"))
        print "12"
        self.verb_trans_hash = pickle.load(open("pickled_files/verb_trans_hash.pk", "rb"))
        print "13"
        self.noun_freq_hash = pickle.load(open("pickled_files/noun_freq_hash.pk", "rb"))
        print "14"
        #self.noun_noun_hash = pickle.load(open("pickled_files/noun_noun_hash.pk", "rb"))
        print "15"
        self.ins_hash = pickle.load(open("pickled_files/ins_hash.pk", "rb"))
        print "Done!"
        self.db = {1: self.visual_thresh_hash, 2: self.hypernym_hash, 3: self.obj_probs, \
                        4: self.mod_ngram_hash, 5: self.plural_hash, 6: self.mod_hash, \
                        7: self.det_hash, 8: self.att_hash, 9: self.preps, 10: self.verb_hash, \
                        11: self.prep_hash, 12: self.verb_trans_hash, 13: self.noun_freq_hash,
                        15: self.ins_hash}

    def read_raw_data(self):
        #dir = "/data/share/corpora_stats/"
        dir = "KB/"
        a = open(dir + "thresholds", "r")
        self.visual_thresh_file = a.readlines()
        a.close()
        b = open(dir + "NYT+WSJ.auto.mod.noOOV", "r")
        self.mod_ngram_file = b.readlines()
        b.close()
        c = open(dir + "flickr.wn_hyps")
        self.wn_hyps = c.readlines()
        c.close()
        d = open(dir + "preps", "r")
        self.prep_file = d.readlines()
        d.close()
        # Swap in NYT data here...closed-class word, so you can use a larger corpus.
        #self.det_file = \
        #open(dir + "flickr_stats/flickr.det_heads", "r").readlines()
        e = open(dir + "nyt_stats/nyt.det_heads", "r")
        self.det_file = e.readlines()
        e.close()
        f = open(dir + "flickr_stats/flickr.mod_nouns", "r")
        self.mod_file = f.readlines()
        f.close()
        g = open(dir + "att_groups", "r")
        self.att_files = g.readlines()
        g.close()
        h = open(dir + "flickr_stats/flickr.verb_nouns", "r")
        self.verb_noun_file = h.readlines()
        h.close()
        i = open(dir + "flickr_stats/flickr.noun_verbs", "r")
        self.noun_verb_file = i.readlines()
        i.close()
        # Swap in NYT data here?
        j = open(dir + "flickr_stats/flickr.verb_preps", "r")
        self.verb_prep_file = j.readlines()
        j.close()
        #self.verb_prep_file = \
        #open(dir + "corpora_stats/redo/nyt.verb_preps", "r").readlines()
        # Swap in a NYT data here
        k = open(dir + "flickr_stats/flickr.prep_nouns", "r")
        self.prep_noun_file = k.readlines()
        k.close()
        #self.prep_noun_file = \
        #open(dir + "corpora_stats/redo/nyt.prep_nouns", "r").readlines()
        l = open(dir + "flickr_stats/flickr.noun_preps", "r")
        self.noun_prep_file = l.readlines()
        l.close()
        #self.noun_prep_file = \
        #open(dir + "corpora_stats/redo/nyt.noun_preps", "r").readlines()
        m = open(dir + "flickr_stats/flickr.verb_transitivity", "r")
        self.verb_trans_file = m.readlines()
        m.close()
        n = open(dir + "flickr_stats/flickr.noun_cooccurrences", "r")
        self.nouns_file = n.readlines()
        n.close()
        o = open(dir + "plurals", "r")
        self.plurals_file = o.readlines()
        o.close()
        p = open(dir + "in_list", "r")
        self.ins_file = p.readlines()
        p.close()
        print "Creating models..."
        print "1"
        self.get_vis_thresh()
        print "2"
        self.read_mod_ngram()
        #self.mod_ngram_hash = pickle.load(open("pickled_files/mod_ngram_hash.pk", "rb"))
        print "3"
        self.read_plurals()
        print "4"
        self.read_det()
        print "5"
        self.read_mod()
        print "6"
        self.read_att()
        print "7"
        self.read_preps()
        print "8"
        self.read_noun_freq()
        print "9"
        self.read_verb_noun()
        print "10"
        self.read_noun_verb()
        print "11"
        self.read_verb_prep()
        print "12"
        self.read_prep_noun()
        print "13"
        self.read_noun_prep()
        print "14"
        self.read_verb_trans()
        #self.read_noun_noun()
        print "15"
        self.read_in_wn()
        print "16"
        self.read_ins()

    def get_as_db(self):
       return self.db

    def read_as_db(self, db):
        self.visual_thresh_hash = db[1]
        self.hypernym_hash = db[2]
        self.obj_probs = db[3]
        self.mod_ngram_hash = db[4]
        self.plural_hash = db[5]
        self.mod_hash = db[6]
        self.det_hash = db[7]
        self.att_hash = db[8]
        self.preps = db[9]
        self.verb_hash = db[10]
        self.prep_hash = db[11]
        self.verb_trans_hash = db[12]
        self.noun_freq_hash = db[13]
        self.ins_hash = db[15]

    def get_vis_thresh(self):
        for line in self.visual_thresh_file:
            split_line = line.split()
            label = split_line[0]
            thresh = float(split_line[1])
            self.visual_thresh_hash[label] = thresh
        if not self.read_pickle:
            pickle.dump(self.visual_thresh_hash, open("pickled_files/visual_thresh_hash.pk", "wb"))

    def read_mod_ngram(self):
        x = 1
        for line in self.mod_ngram_file[8:]:
            inc = False
            line = line.strip()
            if line == "":
                continue
            split_line = line.split("\t")
            if len(split_line) == 1:
                ngram = x
                x += 1
            else:
                words = split_line[1].split()
                score = split_line[0]
                # Speeds up everything by only grabbing ngram evidence for detections we're looking at.
                for word in words:
                    if self.reserved_words != [] and (word not in self.reserved_words) and (word not in ["<s>", "</s>"]):
                        inc = False
                        break
                    inc = True
                if inc:
                    self.mod_ngram_hash[ngram][tuple(words)] = float(score)
        if not self.read_pickle:
            pickle.dump(self.mod_ngram_hash, open("pickled_files/mod_ngram_hash.pk", "wb"))

    def read_plurals(self):
        for line in self.plurals_file:
            split_line = line.split()
            self.plural_hash[split_line[0]] = split_line[1].strip()
        if not self.read_pickle:
            pickle.dump(self.plural_hash, open("pickled_files/plural_hash.pk", "wb"))

    def read_det(self):
        x = 0
        for line in self.det_file:
            split_line = line.split()
            # Mass noun or count noun decision.
            if split_line[0] == "-":
                new_noun = True
                NoDet = False
                no_prob = float(split_line[7])
                plus_split_line = self.det_file[x+1].split()
                yes_prob = float(plus_split_line[7])
                if yes_prob > no_prob:
                    # Then don't store the no prob case.
                    x += 1
                    continue
                else:
                    NoDet=True
            x += 1
            if split_line[0] == "#":
                continue
            w_mod = False
            det_jj = split_line[0]
            if det_jj not in ["+", "-"]:
                split_det_jj = det_jj.split("_")
                det = split_det_jj[0]
                jj = split_det_jj[-1].split("=")
                if jj[-1] == "True":
                    w_mod = True
                elif jj[-1] == "False":
                    w_mod = False
                else:
                    sys.stderr.write("Something weird happened with mods.\n")
                    sys.exit()
            else:
                det = det_jj
            noun_tag_split = split_line[1].split("_")
            noun = "_".join(noun_tag_split[:-1])
            noun_tag = noun_tag_split[-1].upper()
            # Just NNs
            if noun_tag != "NN" and noun_tag != "NNS":
                continue
            try:
                cnt = float(split_line[4])
            except ValueError:
                continue
            if cnt < self.cnt:
                continue
            prob = float(split_line[7])
            if prob < self.word_thresh:
                continue
            elif NoDet and det != "-":
                continue
            try:
                # Without this, this is going to get redefined
                # when a noun has a different tag.  Let's stick
                # with the first one, the highest-scoring one...
                if (det, "DT", w_mod) in self.det_hash[(noun, noun_tag)]:
                    sys.stderr.write("This never happens -- remove.\n")
                    (old_cnt, old_prob) = self.det_hash[(noun, noun_tag)][(det, "DT", w_mod)]
                    # No, let's stick with the tag that has the highest count.
                    if old_cnt > cnt:
                        continue
                    # Otherwise, over-ride the current entry.
                self.det_hash[(noun, noun_tag)][(det, "DT", w_mod)] = (cnt, prob)
            except KeyError:
                self.det_hash[(noun, noun_tag)] = {(det, "DT", w_mod): (cnt, prob)}
        if not self.read_pickle:
            pickle.dump(self.det_hash, open("pickled_files/det_hash.pk", "wb"))

    def read_mod(self):
        for line in self.mod_file:
            split_line = line.split()
            if split_line[0] == "#":
                continue
            noun_tag_split = split_line[0].split("_")
            noun = "_".join(noun_tag_split[:-1])
            noun_tag = noun_tag_split[-1].upper()
            mod_tag_split = split_line[1].split("_")
            mod = "_".join(mod_tag_split[:-1])
            mod_tag = mod_tag_split[-1].upper()
            cnt = float(split_line[4])
            occur = float(split_line[7])
            cooccur = float(split_line[10])
            if cnt < self.cnt:
                continue
            # If this occurs more than just
            # noise level, add it to our knowledge
            # of possible modifiers for this object.
            if cooccur < self.word_thresh:
                continue
            try:
                # Without this, this is going to get redefined
                # when a noun has a different tag.  Let's stick                
                # with the first one, the highest-scoring one...
                if (mod, mod_tag) in self.mod_hash[noun]:
                    continue
                self.mod_hash[noun][(mod, mod_tag)] = (cnt, cooccur, noun_tag)
            except KeyError:
                self.mod_hash[noun] = {(mod, mod_tag): (cnt, cooccur, noun_tag)}
        if not self.read_pickle:
            pickle.dump(self.mod_hash, open("pickled_files/mod_hash.pk", "wb"))

    def read_att(self):
        for line in self.att_files:
            split_line = line.split()
            if line[0] == "#":
                continue
            att = split_line[0]
            self.att_hash[att] = split_line[1:]
        if not self.read_pickle:
            pickle.dump(self.att_hash, open("pickled_files/att_hash.pk", "wb"))
    
    def read_preps(self):
        for line in self.prep_file:
            split_line = line.split("-")
            first_rel = split_line[0]
            first_rel = first_rel.strip()
            split_rel = first_rel.split(" ")
            # main_prep is the basic spatial relation,
            # corresponding to a bunch of possible prepositions.
            main_prep = split_rel[1]
            self.preps[main_prep] = {'ab':[], 'ba':[]}
            for rel in split_line[1:]:
                rel = rel.strip()
                split_rel = rel.split(" ")
                x = split_rel[0]
                prep = split_rel[1]
                y = split_rel[2]
                # "ab" or "ba"
                key = x + y
                self.preps[main_prep][key] += [prep]
        if not self.read_pickle:
            pickle.dump(self.preps, open("pickled_files/preps.pk", "wb"))

    def read_noun_freq(self):
        for line in self.mod_file:
            if "Occurs: " in line:
                split_line = line.split()
                noun_tag_split = split_line[1].split("_")
                noun = "_".join(noun_tag_split[:-1])
                tag = noun_tag_split[-1]
                freq = split_line[3]
                if noun in self.noun_freq_hash:
                    continue
                self.noun_freq_hash[noun] = float(freq)
        if not self.read_pickle:
            pickle.dump(self.noun_freq_hash, open("pickled_files/noun_freq_hash.pk", "wb"))

    def read_verb_noun(self):
        for line in self.verb_noun_file:
            split_line = line.split()
            if line[0] == "#":
                continue
            verb_tag_split = split_line[0].split("_")
            verb = "_".join(verb_tag_split[:-1])
            verb_tag = verb_tag_split[-1].upper()
            if verb in self.light_verbs or verb_tag not in self.present_tense:
                continue
            node = (verb_tag, verb)
            #-- now the second column --#
            noun_tag_split = split_line[1].split("_")
            noun = "_".join(noun_tag_split[:-1])
            noun_tag = noun_tag_split[-1].upper()
            if noun_tag != "NN" and noun_tag != "NNS":
                continue
            try:
                cnt = float(split_line[4])
            except ValueError:
                continue
            # Ignore things with few counts.
            # Do we want to back off to vlc here?
            if cnt < self.cnt:
                continue
            # Can change this to PMI, etc.
            try:
                prob = float(split_line[7])
            except ValueError:  # This happens with odd parses that have spaces...just forget it.
                continue
            if prob < self.word_thresh:
                continue
            # verb_hash = {noun : {verb : {...}}}
            # ordered with noun first.
            # Probability of n occurring after v
            try:
                # Without this, this is going to get redefined
                # when a noun has a different tag.  Let's stick
                # with the first one, the highest-scoring one...
                if node in self.verb_hash['a'][noun]:
                    continue
                self.verb_hash['a'][noun][node] = (cnt, prob, noun_tag)
            except KeyError:
                self.verb_hash['a'][noun] = {node: (cnt, prob, noun_tag)}
        #pickle.dump(self.verb_hash, open("verb_hash.pk", "wb"))

    def read_noun_verb(self):
        for line in self.noun_verb_file:
            split_line = line.split()
            if split_line[0] == "#":
                continue
            noun_tag_split = split_line[0].split("_")
            noun = "_".join(noun_tag_split[:-1])
            noun_tag = noun_tag_split[-1].upper()
            if noun_tag != "NN" and noun_tag != "NNS":
                continue
            verb_tag_split = split_line[1].split("_")
            verb = "_".join(verb_tag_split[:-1])
            # Getting rid of light verbs here;
            # we can just add them ourselves.
            # Also, if the subject can't do it,
            # then the object won't either.
            if verb in self.light_verbs:
                continue
            verb_tag = verb_tag_split[-1].upper()
            if verb_tag not in self.present_tense:
                continue
            node = (verb_tag, verb)
            cnt = float(split_line[4])
            # Ignore things with few counts.
            # Do we want to back off to vlc here?
            if cnt < self.cnt:
                continue
            try:
                prob = float(split_line[7])
            except ValueError:
                continue
            if prob < self.word_thresh:
                continue
            try:
                # Without this, this is going to get redefined
                # when a noun has a different tag.  Let's stick
                # with the first one, the highest-scoring one...
                if node in self.verb_hash['b'][noun]:
                    continue
                self.verb_hash['b'][noun][node] = (cnt, prob, noun_tag)
            except KeyError:
                self.verb_hash['b'][noun] = {node: (cnt, prob, noun_tag)}
        if not self.read_pickle:
            pickle.dump(self.verb_hash, open("pickled_files/verb_hash.pk", "wb"))

    def read_verb_prep(self):
        for line in self.verb_prep_file:
            split_line = line.split()
            if split_line[0] == "#":
                continue
            verb_tag_split = split_line[0].split("_")
            verb = "_".join(verb_tag_split[:-1])
            verb_tag = verb_tag_split[-1].upper()
            if verb_tag not in self.present_tense or verb in self.light_verbs:
                continue
            prep_tag_split = split_line[1].split("_")
            prep = "_".join(prep_tag_split[:-1])
            prep_tag = prep_tag_split[-1].upper()
            node = (prep_tag, prep)
            cnt = float(split_line[4])
            # Ignore things with few counts.
            # Do we want to back off to vlc here?
            if cnt < self.cnt:
                continue
            prob = float(split_line[7])
            if prob < self.word_thresh:
                continue
            try:
                # Without this, this is going to get redefined
                # when a noun has a different tag.  Let's stick
                # with the first one, the highest-scoring one...
                if node in self.prep_hash['verb-prep'][verb]:
                    continue
                self.prep_hash['verb-prep'][verb][node] = (cnt, prob, verb_tag)
            except KeyError:
                self.prep_hash['verb-prep'][verb] = {node: (cnt, prob, verb_tag)}
        #pickle.dump(self.prep_hash, open("prep_hash.pk", "wb"))

    def read_prep_noun(self):
        for line in self.prep_noun_file:
            split_line = line.split()
            if split_line[0] == "#":
                continue
            noun_tag_split = split_line[0].split("_")
            noun = "_".join(noun_tag_split[:-1])
            noun_tag = noun_tag_split[-1].upper()
            if noun_tag != "NN" and noun_tag != "NNS":
                continue
            prep_tag_split = split_line[1].split("_")
            prep = "_".join(prep_tag_split[:-1])
            if prep == "from" or prep == "to" or prep == "of":
                continue
            prep_tag = prep_tag_split[-1].upper()
            node = (prep_tag, prep)
            try:
                cnt = float(split_line[4])
            except ValueError:
                continue
            # Ignore things with few counts.
            # Do we want to back off to vlc here?
            if cnt < self.cnt:
                continue
            try:
                prob = float(split_line[7])
            except ValueError:
                continue
            if prob < self.word_thresh:
                continue
            try:
                # The noun is listed first.
                # Without the following, this is going to get redefined
                # when a noun has a different tag.  Let's stick
                # with the first one, the highest-scoring one...
                if node in self.prep_hash['prep-noun'][noun]:
                    continue
                self.prep_hash['prep-noun'][noun][node] = (cnt, prob, noun_tag)
            except KeyError:
                self.prep_hash['prep-noun'][noun] = {node: (cnt, prob, noun_tag)}
        #pickle.dump(self.prep_hash, open("prep_hash.pk", "wb"))

    def read_noun_prep(self):
        for line in self.noun_prep_file:
            split_line = line.split()
            if split_line[0] == "#":
                continue
            noun_tag_split = split_line[0].split("_")
            noun = "_".join(noun_tag_split[:-1])
            noun_tag = noun_tag_split[-1].upper()
            if noun_tag != "NN" and noun_tag != "NNS":
                continue
            prep_tag_split = split_line[1].split("_")
            prep = "_".join(prep_tag_split[:-1])
            # Directional/state preps, not spatial.  
            if prep == "from" or prep == "to" or prep == "of":
                continue
            prep_tag = prep_tag_split[-1].upper()
            node = (prep_tag, prep)
            try:
                cnt = float(split_line[4])
            except ValueError:
                continue
            # Ignore things with few counts.
            # Do we want to back off to vlc here?
            if cnt < self.cnt:
                continue
            try:
                prob = float(split_line[7])
            except ValueError:
                continue
            if prob < self.word_thresh:
                continue
            try:
                # The noun is listed first.
                # Without the following, this is going to get redefined
                # when a noun has a different tag.  Let's stick
                # with the first one, the highest-scoring one...
                if node in self.prep_hash['noun-prep'][noun]:
                    continue
                self.prep_hash['noun-prep'][noun][node] = (cnt, prob, noun_tag)
            except KeyError:
                self.prep_hash['noun-prep'][noun] = {node: (cnt, prob, noun_tag)}
        if not self.read_pickle:
            pickle.dump(self.prep_hash, open("pickled_files/prep_hash.pk", "wb"))

    def read_verb_trans(self):
        for line in self.verb_trans_file:
            split_line = line.split()
            if split_line[0] == "#":
                continue
            # Pol = whether an object noun follows.
            pol = split_line[0]
            verb = split_line[1]
            cnt = split_line[4]
            if cnt < self.cnt:
                continue
            prob = float(split_line[7])
            # HARD-CODED prob here to decided if verb is
            # transitive or intransitive:  If less than
            # a quarter of the evidence points to (in)transitive,
            # then let's not store it as (in)transitive. 
            # Finesse this as needed.
            if prob < 0.25:
                continue
            try:
                self.verb_trans_hash[verb][pol] = prob
            except KeyError:
                self.verb_trans_hash[verb] = {pol:prob}
        if not self.read_pickle:
            pickle.dump(self.verb_trans_hash, open("pickled_files/verb_trans_hash.pk", "wb"))

    def read_noun_noun(self):
        for line in self.nouns_file:
            split_line = line.split()
            n1_split = split_line[0].split("_")
            n2_split = split_line[1].split("_")
            n1 = "_".join(n1_split[:-1])
            n2 = "_".join(n2_split[:-1])
            npmi = float(split_line[-1])
            self.noun_noun_hash[(n1, n2)] = npmi        
        if not self.read_pickle:
            pickle.dump(self.noun_noun_hash, open("pickled_files/noun_noun_hash.pk", "wb"))

    def read_ins(self):
        for line in self.ins_file:
            line = line.strip()
            split_line = line.split()
            obj = split_line[0]
            in_objs = split_line[1:]
            self.ins_hash[obj] = in_objs
        if not self.read_pickle:
            pickle.dump(self.ins_hash, open("pickled_files/ins_hash.pk", "wb"))


    def get_determiners(self, obj, tag="NN"):
        try:
            return self.det_hash[(obj, tag)]
        except KeyError:
            return self.det_hash[(obj, "NN")]

    def get_mods(self, obj):
        try:
            return self.mod_hash[obj]
        except KeyError:
            return {}

    def get_noun_freq_hash(self, obj):
        try:
            return self.noun_freq_hash[obj]
        except KeyError:
            return 0.0

    def most_prob_word(self, word_hash):
        most_prob = 0.0
        c_x = None
        for x in word_hash:
            prob = word_hash[x]
            if prob > most_prob:
                most_prob = prob
                c_x = x
        return c_x

    def order_mods(self, mods, obj):
        # Orders mods from Mitchell et al. 2011 N-gram model.
        iter_permuts = itertools.permutations(mods)
        permuts = []
        found_order = ()
        max_score = -99999999
        for perm in iter_permuts:
            perm_list = list(perm)
            order_mods = ["<s>"] + perm_list + [obj, "</s>"]
            permuts += [perm_list]
            num = len(order_mods)
            max = num
            min = 0
            score = 0.0
            ordered = []
            last_max = 0
            while ordered != order_mods:
                t_order_mods = tuple(order_mods[min:max])
                # If we've hit the unigram horizon, continue (Pruning hack)
                if max == last_max:
                    break
                if t_order_mods in self.mod_ngram_hash[len(t_order_mods)]:
                    ordered = order_mods[:max]
                    last_max = max
                    score += self.mod_ngram_hash[len(t_order_mods)][t_order_mods]
                    if ordered[-1] == "</s>":
                        if score > max_score:
                            max_score = score
                            found_order = perm_list
                    else:
                        min += 1
                        max = num
                else:
                    max -= 1
        return found_order

    def get_att(self, mod):
        for att in self.att_hash:
            if mod in self.att_hash[att]:
                # If this happens more than 
                # once?
                return att
        return None

    def get_intrans_VP(self, i):
        verb_hash = {}
        try:
            for verb in self.verb_hash['b'][i]:
                verb_hash[verb] = self.verb_hash['b'][i][verb][1]
            return verb_hash
        except KeyError:
            return {}

    def is_verb_intrans(self, verb):
        try:
            # HARD-CODED transitive prob from developing on fake data:
            # Finesse as needed.  If the verb is often
            # intransitive (40% of the time or more),
            # then say it can be used intransitively. 
            if self.verb_trans_hash[verb]["-"] > 0.4:
                return True
        except KeyError:
            return False
        return False


    def get_VPs(self, i, j, PP_js={}):
        """ Change this so the VP hash is actually getting
            sent here -- thus when action detections fire, we're ok.
        """
        # At this point, we just want the set
        # that can go NP -> V and V -> PP OR V -> NP....
        if PP_js != {}:
            verb_prep_hash = {}
            for prep_tuple in PP_js:
                prep = prep_tuple[1]
                try:
                 for verb_tuple in self.verb_hash['b'][i]:
                    verb = verb_tuple[1]
                    # From the set of prepositions leading up to 
                    # the noun, grab the prepositions that follow this verb,
                    # And get the probabilities; also get the probability
                    # of this noun following the verb.
                    if prep_tuple in self.prep_hash['verb-prep'][verb]:
                        prep_prob = self.prep_hash['prep-noun'][j][prep_tuple][1]
                        verb_prob = self.verb_hash['b'][i][verb_tuple][1]
                        verb_prep_hash[(prep_tuple, verb_tuple)] = round(math.sqrt(prep_prob * verb_prob), 4)
                except KeyError:
                    pass
            return verb_prep_hash
        (unk_i, unk_j) = (False, False)
        verb_list_i = {}
        verb_list_j = {}
        try:
            verb_list_i = self.verb_hash['b'][i]
        except KeyError:
            unk_i = True
        try:
            verb_list_j = self.verb_hash['a'][j]
        except KeyError:
            unk_j = True
        if unk_i or unk_j:
            # This should compute a whole bunch of
            # spatial relations, actually.
            # Just returns nothing for now.
            return {}
        verb_hash = {}
        for verb_tuple in verb_list_i:
            i_prob = verb_list_i[verb_tuple][1]
            if verb_tuple in verb_list_j:
                # Combining probs.
                j_prob = verb_list_j[verb_tuple][1]
                verb_hash[verb_tuple] = round(math.sqrt(i_prob * j_prob), 4)
        return verb_hash

    def get_preps(self, prep, order):
        return self.preps[prep][order]

    def get_PPs(self, i, j=None, c_preps=None):
        prep_hash = {}
        # For each of the verbs we're considering for
        # an obj complement, see if that verb can also appear with a PP.
        # (e.g., boy sees store / boy sees in a store)
        if j == None:
            if c_preps == None:
                return self.prep_hash['prep-noun'][i]
            else:
                # If we have no preps in our corpus data
                # between these words at this threshold,
                # just use the visual data verbatim.
                # Otherwise, grab the intersection of what
                # the vision guesses and what we've seen in our corpus.
                try:
                    for o_prep in self.prep_hash['prep-noun'][i]:
                        for c_prep in c_preps:
                            if c_prep == o_prep[1]:
                                prep_hash[o_prep] = self.prep_hash['prep-noun'][i][o_prep][1]
                            else: # Just ignore it.
                                pass
                except KeyError:
                    # Not in our corpus:  Make all the vision preps 
                    # equiprobable.
                    for c_prep in c_preps:
                        prep_hash[c_prep] = self.word_thresh
                return prep_hash
        # Gets the intersection of:
        # 1.  Prepositions the vision system suggests
        # 2.  Prepositions that head a PP complement of the subject noun.
        # 3.  Prepositions that head a PP complemented by an NP headed by the object noun.
        try:
            for node in self.prep_hash['prep-noun'][j]:
                if node in self.prep_hash['noun-prep'][i]:
                    i_prob = self.prep_hash['noun-prep'][i][node][1]
                    j_prob = self.prep_hash['prep-noun'][j][node][1]
                    ij_prob = round(math.sqrt(i_prob * j_prob), 4)
                    if c_preps != None:
                        # Vision system can confuse "in" with "against" or
                        # "in front of" in a 2-d image.  Fixes this issue when 
                        # the language model suggests "against".
                        if node[1] in c_preps or (node[1] == "against" and "in" in c_preps):
                            prep_hash[node] = ij_prob #j_prob
                        else:
                            # Prep is not in c_preps; ignore it.
                            pass
                    else:
                        prep_hash[node] = ij_prob
        except KeyError:
            pass # When some NN or NNS is not in our database.
        return prep_hash

    def get_noun_noun(self, n1, n2):
        n_list = [n1, n2]
        n_list.sort()
        try:
            return self.noun_noun_hash[tuple(n_list)]
        except KeyError:
            return None

    def read_in_wn(self):
        for line in self.wn_hyps:
            if line[0] == "#":
                continue
            line = line.strip()
            split_line = line.split()
            num = int(split_line[0])
            pos = int(split_line[1])
            prob = float(split_line[-1])
            hypernym = split_line[2]
            try:
                self.hypernym_hash[num][hypernym][pos] = prob
            except KeyError:
                try:
                    self.hypernym_hash[num][hypernym] = {pos:prob}
                except KeyError:
                    self.hypernym_hash[num] = {hypernym:{pos:prob}}
        if not self.read_pickle:
            pickle.dump(self.hypernym_hash, open("pickled_files/hypernym_hash.pk", "wb"))

    def get_hyp_probs(self, obj_set):
        self.num_objs = len(obj_set)
        found = False
        for obj in obj_set:
            if obj in self.obj_probs:
                continue
            obj_syn = wn.synset(obj + ".n.01")
            found = False
            for hypernym_path in obj_syn.hypernym_paths():
                if found:
                    break
                hypernym_path = list(hypernym_path)
                hypernym_path.reverse()
                for hypernym in hypernym_path:
                    if found:
                        break
                    # Return p(pos|num_objs) for all possible pos.
                    # e.g., {1:0.4, 2:.3, 3:.3}
                    self.obj_probs[obj] = self.hypernym_hash[self.num_objs][str(hypernym)]
                    found = True
                    break
        if not self.read_pickle:
            pickle.dump(self.obj_probs, open("pickled_files/obj_probs.pk", "wb"))

    def order_set(self, obj_list):
        final_set = []
        # For each position...
        tuple_list = []
        for i in range(1, self.num_objs + 1):
            # Grab the probability of each object in this position.
            for obj in self.obj_probs:
                try:
                    obj_prob = self.obj_probs[obj][i]
                except KeyError:
                    obj_prob = .000000000001
                tuple_thing = (obj_prob, obj, i)
                tuple_list += [tuple_thing]
        tuple_list.sort()
        tuple_list.reverse()
        order_hash = {}
        for tuple_thing in tuple_list:
            obj = tuple_thing[1]
            pos = tuple_thing[2]
            if pos not in order_hash:
                while obj in obj_list:
                    try:
                        order_hash[pos][obj] += 1
                    except KeyError:
                        try:
                            order_hash[pos][obj] = 1
                        except KeyError:
                            order_hash[pos] = {obj:1}
                    obj_index = obj_list.index(obj)
                    obj_list.pop(obj_index)
        for pos in sorted(order_hash):
            obj = order_hash[pos].keys()
            obj = obj[0]
            for i in range(order_hash[pos][obj]):
                final_set += [obj]
        # Chosen object is the most probable one.
        return final_set

    def cluster_objs(self, obj_list):
        """ Nominal ordering from hypernyms. """
        self.get_hyp_probs(obj_list)
        final_set = self.order_set(obj_list)
        return final_set
