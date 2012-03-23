import sys
import re
import itertools
import yaml
import pickle
from copy import copy
from queryKB import queryKB
from nltk.corpus import wordnet as wn
from math import log

class Sofie():
    def __init__(self, KB_obj, data={}, word_thresh=.01, count_cutoff=2, vision_thresh=.3, spec_post=False, halluc_set=[], with_preps=True, pickled=True):
        """ Input:  Word_thresh:  Likelihood cutoff for beam of (tag, word) pairs selected by noun anchor 
                    Count_cutoff:  Raw count cutoff for collected co-occurrences
                    Vision_thresh:  Blanket threshold at which to ignore vision detections 
                    spec_post:  When generating for one specific image
                    halluc_set:  Set of syntactic categories being 'hallucinated' from language modelling alone. 
                    with_preps:  if True, use basic spatial relations from vision bounding boxes to guide
                                prep selection, otherwise guess preps from language modelling alone. """
        self.data = data
        self.animal = ""
        self.detections = {}
        self.mod_detections = {}
        self.action_detections = {}
        self.prep_detections = {}
        self.word_thresh = word_thresh
        self.count_cutoff = count_cutoff
        self.vision_thresh = vision_thresh
        self.halluc_set = halluc_set
        # Colors and actions the system is currently detecting, mapped to their progressive forms.
        # Small set, so hard-wired for now.  
        self.colors = ('red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'black', 'white')
        # Not utilized in system demo.
        self.actions = {'stand':'standing', 'swim':'swimming', 'sit':'sitting', 'face':'facing', 'run':'running', 'fly':'flying', 'liedown':'lying down', 'standeat':'standing and eating'}
        # Stores all of the corpus-based probability estimates, and functions to grab them.
        self.KB_obj = KB_obj
        # Present tense verb forms.
        self.present_tense = ("VBG", "VBZ", "VBN")
        self.label_id_hash = {}
        self.pickled = pickled
        self.spec_post = spec_post
        self.with_preps = with_preps
        ## STEP 1.
        self.get_detections()

    def order_by_frequency(self, input_list):
        """ Orders nouns by how frequent they are. """
        obj_list = []
        for obj in input_list:
            freq = self.KB_obj.get_noun_freq_hash(obj)
            obj_list += [(freq, obj)]
        obj_list.sort()
        output_list = []
        for obj_tuple in obj_list:
            obj = obj_tuple[1]
            output_list += [obj]
        return output_list   

    # ---- Surface Realization Functions ---- #

    def __surface_node__(self, node):
        """ Pretty string realization of a node. """
        surface_str = "(" + node[0]
        x = 1
        while x < len(node):
            sub_node = node[x]
            if isinstance(sub_node, tuple):
                surface_str += " " + self.__surface_node__(sub_node)
            else:
                surface_str += " " + node[x] + " " + str(node[x+1])
                x += 1
            x += 1
        surface_str += ")" 
        return surface_str

    def __get_det__(self, det, word_tag, prob):
        """ Indefinite article surface form affected by following word: adjust
            surface form accordingly. """
        word = word_tag[1]
        if det == "an":
            if word[0] not in "aeiou":
                det = "a"
        elif det == "a":
            if word[0] in "aeiou":
                det = "an"
        return ("DT", det, prob)

    def get_NP(self, post_id, id_n, obj, is_plural, det_hash, mod_hash):
        """ Creates a noun phrase from the given detections. """
        att_hash = {}
        NPs = {}
        mods = {}
        if obj not in is_plural:
            o_tag = "NN"
            ### STEP 5:  Limits adjectives to the set that are not
            ### mutually exclusive.
            # Modifier:  Using Vision/Language intersection
            # and the M.E. hypothesis
            for mod in self.mod_detections[post_id][id_n]:
                # These are just JJ for now; don't actually come into play
                # but I'm saving this so it can be changed later.
                mod_tag = "JJ"
                # Later, we can double-check for stuff
                # based on what's expected...
                # Choose the intersection between detected mods
                # and language mods.
                att = self.KB_obj.get_att(mod)
                v_score = float(self.mod_detections[post_id][id_n][mod])
                if v_score < self.vision_thresh:
                    continue
                # Until we have atts for everything,
                # the attribute of a value can be
                # equal to the value
                # M.E. hypothesis
                if att == None:
                    att_hash[mod] = (mod, v_score, mod_tag)
                elif att in att_hash:
                    c_score = att_hash[att][1]
                    # This goes by the vision score.
                    if v_score > c_score:
                        att_hash[att] = (mod, v_score, mod_tag)
                else:
                    att_hash[att] = (mod, v_score, mod_tag)
        else:
            o_tag = "NNS"
        for att in att_hash:
            mod = att_hash[att][0]
            for mod_tuple in mod_hash:
                # Mod is supported by both the vision
                # and the language.
                language_mod = mod_tuple[0]
                prob = mod_hash[mod_tuple][1]
                if mod == language_mod:
                    mods[mod] = prob
                    # Won't hit another -- this saves
                    # it from iterating uselessly.
                    break
        mod_len = len(mods)
        mod_orders = {}
        ### STEP 10:  Order selected modifiers.
        # 1 or more modifiers; orders them using Mitchell et al. 2011 N-gram model.
        while mod_len > 0:
            mod_combinations = itertools.combinations(mods, mod_len)
            mod_len -= 1
            for mod_combination in mod_combinations:
                ordered_mods = self.KB_obj.order_mods(mod_combination, obj)
                new_ordered_mods = []
                for mod in ordered_mods:
                    mod_node = (mod_tag, mod)
                    mod_prob = mods[mod]
                    mod_node_w_prob = (mod_tag, mod, mod_prob)
                    new_ordered_mods += [mod_node_w_prob]
                mod_orders[tuple(new_ordered_mods)] = {}
        for det_tuple in det_hash:
            # Just separates determiners into 'definite' and 'indefinite'.
            det = det_tuple[0]
            det_prob = det_hash[det_tuple][1]
            # d_tag = det_tuple[1]  <-- Only important if we consider dets other than "DT"
            # for estimating p(definite|jj)
            jj_present = det_tuple[2]
            # Presence of article, not a specific one -- skip
            if det == "+":
                continue
            obj_node = (o_tag, obj)
            obj_node_w_prob = (o_tag, obj, 1.0)
            if not jj_present:
                NP = (self.__get_det__(det, obj_node, det_prob), obj_node_w_prob)
                NPs[NP] = {}
            # det == "-" is the null determiner
            if (jj_present or det == "-") and mods != []:
                # Add to the NP list NPs with 
                # all the possible realizations
                # of the modifiers.
                for mod_order in mod_orders:
                    if mod_order == ():
                        continue
                    # If it's an indefinite determiner,
                    # make sure you grab 'a' or 'an' 
                    # based on the mod that follows it.
                    NP = tuple([self.__get_det__(det, mod_order[0], det_prob)] + list(mod_order) + [obj_node_w_prob]) 
                    NPs[NP] = {}
        return NPs

    def __prenom_or_postnom__(self, nodes):
        """ Surface realization function: selects whether given mods
            should be realized as a prenominal modifier or a postnominal
            modifier.  Currently just handles person color, which tends to
            be clothing.  """
        head_in = False
        new_nodes = []
        if nodes[-1][1] == "person":
            for node in nodes[:-1]:
                if node[0] == "JJ":
                    if node[1] in self.colors:
                        new_nodes += [nodes[-1], ("PP", ("IN", "in", 1.0), ("ADJP", node))]
                        head_in = True
                    else:
                        new_nodes += [tuple(node)]
                else:
                    new_nodes += [tuple(node)]
            if not head_in:
                new_nodes += [tuple(nodes[-1])]
        else:
            new_nodes = nodes
        return tuple(new_nodes)

    def __nonterm_surface__(self, tag, nodes):
        """ Surface realization of non-terminal node. """
        nodes = self.__prenom_or_postnom__(nodes)
        s = " (" + tag
        for node in nodes:
            s += " " + self.__surface_node__(node)
        s += ")"
        return s

    def __nonterm_surface_rels__(self, RELS, cur_str):
        """ Surface realization of embedded non-terminal.
            input:  cur_str = the derived tree headed
                    by an NN. 
                    RELS = verb/prep/conj relations
                           selected by that NN and
                           embedded NN """
        last_rel_tuple = None
        for rel_tuple in RELS:
            # Adjunction operation preserving flatter structures.
            if rel_tuple in ("VP-VBG", "VP-VBN", "VP-VBZ"):
                last_rel_tuple = "VP"
                if "VP" in cur_str:
                    if "VP-VBZ" in cur_str:
                        cur_str = re.sub("\(VP", r"(VP (VP", cur_str)
                        cur_str += " (" + rel_tuple
                    # S -> NP VP 
                    elif rel_tuple == "VP-VBZ":
                        cur_str = " (S" + cur_str + " (" + rel_tuple
                    else:
                        cur_str = " (NP" + cur_str + " (" + rel_tuple
                else:
                    cur_str = " (NP" + cur_str + " (" + rel_tuple
                continue
            elif rel_tuple == "PP":
                if last_rel_tuple == "VP":
                    cur_str += " (" + rel_tuple
                else:
                    cur_str = " (NP" + cur_str + " (" + rel_tuple
                last_rel_tuple = None
                continue
            # Coordination constraint -- triggers mother NP with 
            # NP daughters.
            elif rel_tuple == "CONJP":
                cur_str = " (NP" + cur_str
                last_rel_tuple = None
                continue
            cur_str += " " + self.__surface_node__(rel_tuple)
        return cur_str

    def print_sentence(self, NP1, RELS, NP2, RELS2="", NP3=""):
        """ Generates tree structures. """
        # NP -> NP VP[VBG], NP -> NP VP[VBN], NP -> NP PP, S -> NP VP[VBZ]
        # First object, the subject, realized as NP with
        # prenominal modifiers 
        final_str = self.__nonterm_surface__("NP", NP1)
        # Second object realized in VP, PP,
        # or as a backoff, coordinated with the subject.
        final_str = self.__nonterm_surface_rels__(RELS, final_str)
        final_str += self.__nonterm_surface__("NP", NP2)
        final_str += (")" * (final_str.count("(") - final_str.count(")")))
        if NP3 == "":
            pass
        else:
            final_str = self.__nonterm_surface_rels__(RELS2, final_str)
            final_str += self.__nonterm_surface__("NP", NP3)
            final_str += (")" * (final_str.count("(") - final_str.count(")")))
        return final_str

    def print_sentence_single(self, NP, VP=None):
        """ Generates tree structure for a single object. """
        final_str = self.__nonterm_surface__("NP", NP)
        if VP:
            final_str = self.__nonterm_surface_rels__(VP, final_str)
        final_str += (")" * (final_str.count("(") - final_str.count(")")))
        return final_str

    # ---- Microplanning functions ---- #

    def get_detections(self, data={}):
        """ Takes vision output, places nouns into nodes where
            the language-based constraints can operate. """
        label = ""
        if data == {}:
            data = self.data
        for a in data:
            last_label = label
            try:
                # If this word is stored in our language model as another word,
                # (e.g., motorbike --> motorcycle), change the word identity.
                label = self.KB_obj.label_hash[a['label']]
            except KeyError:
                label = a['label']
            score = a['score']
            # Uncomment in DEV:
            # If this is a low-scoring detection, ignore it.
            # try:
            #    if score < KB_obj.visual_thresh_hash[label]:
            #        continue
            # except KeyError:
            #    pass
            type_n = a['type']
            id_n = a['id']
            post_id = a['post_id']
            # Generating for just a single image.
            if self.spec_post and post_id != self.spec_post:
                continue
            try:
                self.label_id_hash[post_id][id_n] = label
            except KeyError:
                self.label_id_hash[post_id] = {id_n:label}
            try:
                if a['preps'] == {}:
                    self.prep_detections[post_id] = {}
                for id_set in a['preps']:
                    ids = id_set.split(",")
                    id1 = int(ids[0].strip("'"))
                    id2 = int(ids[1].strip("'"))
                    try:
                        self.prep_detections[post_id][(id1, id2)] = a['preps'][id_set]
                    except KeyError:
                        self.prep_detections[post_id] = {(id1, id2): a['preps'][id_set]}
            except KeyError:
                pass
            if type_n == 2:
                # Format assumption:  In vision output,
                # action detection for an object
                # follows that object detection.
                tmp = label.split(last_label)
                label = last_label
                action = tmp[-1]
                try:
                    self.action_detections[post_id][id_n][action] = score
                except KeyError:
                    try:
                        self.action_detections[post_id][id_n] = {action:score}
                    except KeyError:
                        self.action_detections[post_id] = {id_n:{action:score}}
                continue 
            try:
                self.detections[post_id][id_n] = score
                self.mod_detections[post_id][id_n] = {}
            except KeyError:
                self.detections[post_id] = {id_n:score}
                self.mod_detections[post_id] = {id_n:{}}
            try:
                for mod in a['attrs']:
                    self.mod_detections[post_id][id_n][mod] = a['attrs'][mod]
            except KeyError:
                self.mod_detections[post_id][id_n] = {}

    def generate_sentences(self, NPs, obj2_relations={}):
        """ Creates all the trees from the selected constraints. """
        sentence_hash = {}
        if obj2_relations == {}:
            sys.stderr.write("Nothing defined other than an NP.\n")
        else:
            for id_list in obj2_relations:
                sentence_hash[id_list] = {}
                if len(id_list) == 1:
                    id_n = id_list[0]
                    for NP1 in NPs[id_n]:
                        # If we're hallucinating an intransitive verb...
                        if "verb" in self.halluc_set:
                            for RELS in obj2_relations[id_list][(id_n,)]:
                                final_string = self.print_sentence_single(NP1, RELS)
                                sentence_hash[id_list][final_string] = {}
                        else:
                            final_string = self.print_sentence_single(NP1)
                            sentence_hash[id_list][final_string] = {}
                else:
                    mentioned_objs = {}
                    # Note that the way we're traversing the
                    # object list as a declarative sentence,
                    # with obj1 as the subject:
                    # The first item is mentioned
                    # followed by all the other items
                    id3 = ""
                    try:
                        [id1, id2, id3] = id_list[:3]
                    except ValueError:
                        [id1, id2] = id_list[:2]
                    for NP1 in NPs[id1]:
                        for NP2 in NPs[id2]:
                            if id3 != "":
                                for NP3 in NPs[id3]:
                                    for RELS2 in obj2_relations[id_list][(id1, id2)]:
                                        for RELS3 in obj2_relations[id_list][(id1, id3)]:
                                            final_string = self.print_sentence(NP1, RELS2, NP2, RELS3, NP3)
                                            sentence_hash[id_list][final_string] = {}
                            else:
                                for RELS in obj2_relations[id_list][(id1, id2)]:
                                    final_string = self.print_sentence(NP1, RELS, NP2)
                                    sentence_hash[id_list][final_string] = {}
        #print "Now have", sentence_hash
        return sentence_hash

    def check_plurals(self, objs):
        is_plural = {}
        obj_hash_in = {}
        obj_hash_out = {}
        for obj in objs:
            obj_hash_in[obj] = obj_hash_in.setdefault(obj, 0) + 1
        for obj in obj_hash_in:
            if obj_hash_in[obj] > 1:
                obj_hash_out[obj] = self.KB_obj.plural_hash[obj]
                is_plural[self.KB_obj.plural_hash[obj]] = {}
            else:
                obj_hash_out[obj] = obj
        return (obj_hash_out, is_plural)

    def maximize_det_prob(self, dets_with_scores):
        """ Selects the most likely determiners, conditioned on
            presence of an adjective. """
        adj_det_list = []
        noadj_det_list = []
        for det_tuple in dets_with_scores:
            det = det_tuple[0]
            if det == "+":
                continue
            adj = det_tuple[2]
            prob = dets_with_scores[det_tuple][1]
            if adj:
                adj_det_list += [(prob, det_tuple, dets_with_scores[det_tuple])]
            else:
                noadj_det_list += [(prob, det_tuple, dets_with_scores[det_tuple])]
        adj_det_list.sort()
        noadj_det_list.sort()
        adj_det_list.reverse()
        noadj_det_list.reverse()
        det_hash = {}
        if adj_det_list != []:
            det_hash[adj_det_list[0][1]] = adj_det_list[0][2]
        if noadj_det_list != []:
            det_hash[noadj_det_list[0][1]] = noadj_det_list[0][2]
        return det_hash

    def run(self):
        final_sentence_hash = {}
        # For each image..
        for post_id in self.detections:
            final_sentence_hash[post_id] = {}
            # Get the detected objects.
            objs = self.label_id_hash[post_id].values()
            obj_list = []
            is_plural = {}
            # Simplest case:  Only 1 object detected.
            if len(objs) == 1:
                # Figure out just the determiner/action for that guy.
                obj_list = objs
                id_list = self.label_id_hash[post_id].keys()
            else:
                ### STEP 2: Cluster the similar guys together goes here.
                # Get the plural form if there's more than one object
                # of the same type.
                (obj_plural_hash, is_plural) = self.check_plurals(objs)
                ### STEP 3: Order nouns within each group.
                obj_list = self.KB_obj.cluster_objs(obj_plural_hash.keys())
                id_list = []
                # These should really be (obj, id) tuples returned from the
                # KB function, this is a messy way of handling this.
                for obj in obj_list:
                    for id_x in self.label_id_hash[post_id]:
                        if self.label_id_hash[post_id][id_x] == obj and id_x not in id_list:
                            id_list += [id_x]
                            # This means we're just storing a random index
                            # for a pluralized group of objects.
                            break
                tmp_obj_list = []
                for obj in obj_list:
                    tmp_obj_list += [obj_plural_hash[obj]]
                obj_list = tmp_obj_list 
            ### STEP 4:  Create all tree structures.   
            # Initialize our NPs hash
            NPs = {}
            # Stores the possible modifiers for each object
            obj_mod_hash = {}
            # Stores the possible determiners for each object
            obj_det_hash = {}
            VPs = {}
            PPs = {}
            CONJPs = {}
            if self.with_preps:
                # Prepositions returned from the vision bounding boxes
                given_preps = self.prep_detections[post_id]
            obj_id_cnt = 0
            while obj_id_cnt < len(id_list):
                id_n = id_list[obj_id_cnt]
                obj = obj_list[obj_id_cnt]
                obj_id_cnt += 1
                # Figure out if we should treat this as
                # a mass noun or count noun, 'a' or 'an', etc.
                if obj not in obj_det_hash:
                    if obj in is_plural:
                        obj_det_hash[obj] = self.KB_obj.get_determiners(obj, "NNS")
                    else:
                        obj_det_hash[obj] = self.KB_obj.get_determiners(obj)
                    # Selects the most likely determiners:
                    # 1 - When there's an adjective
                    # 2 - When there's not an adjective
                    obj_det_hash[obj] = self.maximize_det_prob(obj_det_hash[obj])
                # Figure out if we should attach
                # any modifiers
                if obj not in obj_mod_hash:
                    if obj in is_plural:
                        obj_mod_hash[obj] = {}
                    else:
                        obj_mod_hash[obj] = self.KB_obj.get_mods(obj)
                # Return all NPs with determiners + adjectives
                # that combine with the head noun above the word threshold.
                NPs[id_n] = self.get_NP(post_id, id_n, obj, is_plural, obj_det_hash[obj], obj_mod_hash[obj])
            sentences = {}
            n = 0
            # Now we have the basics for each individual NP;
            # Figure out the relations between NPs (VP, PP, or VP PP).
            if len(obj_list) == 1:
                obj = obj_list[0]
                # Checks if the vision system has provided action detections
                # or if hallucinated verbs have been requested.
                try:
                    for action in self.action_detections[post_id][id_n]:
                        VPs[(id_n,)][("VBG", actions[action])] = 1.0
                except KeyError:
                    if "verb" in self.halluc_set:
                        VPs[(id_n,)] = self.KB_obj.get_intrans_VP(obj)
                    else:
                        VPs[(id_n,)] = {}
                obj2_relations = {}
                sentences[n] = (id_n,)
            else:
                mentioned = {}
                x = 0
                last_y = 0
                V_PPs = {}
                while x < len(id_list):
                    i = id_list[x]
                    obj_i = obj_list[x]
                    x += 1
                    y = x
                    if i in mentioned:
                        continue
                    mentioned[i] = {}
                    while y < len(id_list):
                        j = id_list[y]
                        obj_j = obj_list[y]
                        # Should never happen, but just in case code gets changed...
                        if j in mentioned:
                            continue
                        y += 1
                        mentioned[j] = {}
                        c_preps = None
                        if self.with_preps:
                            try:
                                g_prep = given_preps[(i, j)]
                                c_preps = self.KB_obj.get_preps(obj_i, g_prep, obj_j, 'ab')
                            except KeyError:
                                g_prep = given_preps[(j, i)]
                                c_preps = self.KB_obj.get_preps(obj_j, g_prep, obj_i, 'ba')
                        ### STEP 8:  Likely verbs requested, generate VP structures for them.
                        # Generates VP --> V NP and VP --> V PP structures
                        if "verb" in self.halluc_set:
                            VPs[(i, j)] = self.KB_obj.get_VPs(obj_i, obj_j)
                            PP_js = self.KB_obj.get_PPs(obj_j, None, c_preps)
                            V_PPs[(i, j)] = self.KB_obj.get_VPs(obj_i, obj_j, PP_js)
                        else:
                            VPs[(i, j)] = {}
                            V_PPs[(i, j)] = {}
                        ### STEP 6:  Create all trees that combine at the PP level.
                        PPs[(i, j)] = self.KB_obj.get_PPs(obj_i, obj_j, c_preps)
                        # Backoff:  Cannot find a way for the objects to go together
                        # with a preposition, so we use a conjunction.
                        if PPs[(i, j)] == {}:
                            CONJPs[(i, j)] = {("CC", "and"): 1.0}
                        if y == 3 or (y - last_y) == 3:
                            sentences[n] = (id_list[x-1:y])
                            x = y
                            last_y = y
                            n += 1
                            y = len(id_list)
                last_items = id_list[last_y:]
                if last_items != []:
                    if len(last_items) == 1:
                        id_n = last_items[0]
                        obj = self.label_id_hash[post_id][id_n]
                        try:
                            for action in self.action_detections[post_id][id_n]:
                                VPs[(id_n,)][("VBG", actions[action])] = 1.0
                        except KeyError:
                            if "verb" in self.halluc_set:
                                VPs[(id_n,)] = self.KB_obj.get_intrans_VP(obj)
                        PPs[(id_n,)] = {}
                    sentences[n] = (id_list[last_y:])
            ### STEP 9:  Generating trees following grammar until all object nouns
            ### are accounted for.
            s_count = 0
            obj_relations = {}
            for sentence in sentences:
                id_list = sentences[sentence]
                s = tuple(id_list)
                obj_relations[s] = {}
                s_count += 1
                i = 0
                try:
                    id_tuples = [(id_list[0], id_list[1])]
                    try:
                        id_tuples += [(id_list[0], id_list[2])]
                    except IndexError:
                        pass
                except IndexError:
                    id_tuples = [(id_list[0],)]
                #print "id tuple is", id_tuples
                for id_tuple in id_tuples:
                    id1 = id_tuple[0]
                    try:
                        id2 = id_tuple[1]
                    except IndexError:
                        pass
                    chose_prep = False
                    # Relation leading up to object is initialized
                    obj_relations[s][id_tuple] = {}
                    if "verb" in self.halluc_set:
                        if id_tuple in VPs and VPs[id_tuple] != {}:
                            for verb_node in VPs[id_tuple]:
                                # For each of the verbs you can use,
                                # get the prob of that (obj, verb, obj)
                                # triple
                                verb_obj_prob = VPs[id_tuple][verb_node]
                                tag_verb_prob = (verb_node[0], verb_node[1], verb_obj_prob)
                                obj_relations[s][id_tuple][("VP-" + verb_node[0], tag_verb_prob)] = {}
                        # Rules that take NP complements:  Only fire when there's more than 1 NP.
                        if len(id_tuple) > 1:
                            if id_tuple in V_PPs:
                                for prep_verb_tuple in V_PPs[id_tuple]:
                                    prep_verb_obj_prob = V_PPs[id_tuple][prep_verb_tuple]
                                    verb_tuple = prep_verb_tuple[1]
                                    tag_verb_prob = (verb_tuple[0], verb_tuple[1], prep_verb_obj_prob)
                                    prep_tuple = prep_verb_tuple[0]
                                    tag_prep_prob = (prep_tuple[0], prep_tuple[1], prep_verb_obj_prob)
                                    obj_relations[s][id_tuple][("VP-" + verb_tuple[0], tag_verb_prob, "PP", tag_prep_prob)] = {}
                    if len(id_tuple) > 1:
                        # Backoff: When no object relations can be generated,
                        # just say 'and'.
                        if PPs[id_tuple] == {}:
                            for conj_node in CONJPs[id_tuple]:
                                prob = CONJPs[id_tuple][conj_node]
                                tag_conj_prob = (conj_node[0], conj_node[1], prob)
                                #print tag_conj_prob
                                obj_relations[s][id_tuple][("CONJP", tag_conj_prob)] = {}
                        else:
                            for prep_node in PPs[id_tuple]:
                                # As well as the preposition alone
                                # (for generating NPs instead of
                                # a full sentence)
                                prob = PPs[id_tuple][prep_node]
                                tag_prep_prob = (prep_node[0], prep_node[1], prob)
                                obj_relations[s][id_tuple][("PP", tag_prep_prob)] = prob
                # Generate sentences.
                print "Generating with", NPs, obj_relations
                final_sentence_hash[post_id] = self.generate_sentences(NPs, obj_relations)
                print "*** final sentence is", final_sentence_hash[post_id]
        return final_sentence_hash

def print_usage():
    sys.stderr.write("Usage:  python generate.py (data_file|--post-id=2008_XXXXXX.txt) [OPTIONS]\n")
    sys.stderr.write("### Options ###\n")
    sys.stderr.write("--word-thresh=X\t\t\tBlanket probability threshold for word-coocurrence/lpcfg rules.\n")
    sys.stderr.write("--post-id=X\t\t\tPrints out descriptions for a specific post-id (if available from read in output).\n")
    sys.stderr.write("--hallucinate=[verb|noun|adj]\t'Hallucinates' open class things.  Currently just supports verbs.\n")
    sys.stderr.write("--count-cutoff=X\t\tHow many times something must be observed before considering it as an option.\n")
    sys.stderr.write("--with-preps\t\t\tInput includes a preposition specification, and output will be constrained in accordance\n\t\t\t\t(calculated from bounding box; true for BabyTalk input).\n")
    sys.stderr.write("--not-pickled\t\t\tDo not read in saved pickle files.\n")
    sys.exit()

if __name__ == "__main__":
    word_thresh = .01
    count_cutoff = 2
    vision_thresh = .3
    spec_post = False
    halluc_set = []
    with_preps = True
    pickled = True
    if len(sys.argv) < 2:
        print_usage()
    check_arg_1 = sys.argv[1].split("=")
    # Makes it possible to input a whole data file
    # or 1 specific image.
    if check_arg_1[0] == "--post-id":
        spec_post = check_arg_1[1]
    for arg in sys.argv[2:]:
        split_arg = arg.split("=")
        if split_arg[0] == "--word-thresh":
            word_thresh = float(split_arg[1])
        elif split_arg[0] == "--post-id":
            spec_post = split_arg[1]
        elif split_arg[0] == "--hallucinate":
            halluc_set += [(split_arg[1])]
        elif split_arg[0] == "--count-cutoff":
            count_cutoff = int(split_arg[1])
        elif split_arg[0] == "--with-preps":
            if split_arg[1] == "True":
                with_preps = True
            elif split_arg[1] == "False":
                with_preps = False
        elif split_arg[0] == "--not-pickled":
            pickled = False
        elif split_arg[0] == "--vision-thresh":
            vision_thresh = float(split_arg[1])
        else:
            print_usage()
    # Commented out stuff below is from
    # different input approaches we've used...
    #if not pickled:
    #data = yaml.load(file(sys.argv[1], 'r'))
    #pickle.dump(data, open("pickled_files/data.pk", "wb"))
    ##else:
    if len(check_arg_1) == 1:
        data = pickle.load(open("pickled_files/data.pk", "rb"))
    else:
        data = yaml.load(file("vision_out-dev/" + spec_post, 'r'))

    if not pickled:
        KB_obj = queryKB(word_thresh, count_cutoff, False)
    else:
        KB_obj = queryKB(word_thresh, count_cutoff)
    sofie_obj = Sofie(KB_obj, data, word_thresh, count_cutoff, vision_thresh, spec_post, halluc_set, with_preps, pickled)
    #sofie_obj.get_detections()
    final_sentence_hash = sofie_obj.run()
    for post_id in sorted(final_sentence_hash):
        print "***", post_id
        for s_num in final_sentence_hash[post_id]:
            for sentence in final_sentence_hash[post_id][s_num]:
                print sentence
