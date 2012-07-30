#!/usr/bin/python

import os, string, sys
import re
from Cookie import SimpleCookie
from os import path
from generate import Sofie
from queryKB import queryKB
import tornado.ioloop
import tornado.web

bin_path = path.join("./","")
image_path = path.join("static/", "")
vision_path = path.join("vision_out-dev/", "")
KB_obj = queryKB()

class MainHandler(tornado.web.RequestHandler, tornado.web.UIModule):
    def initialize(self, database):
        """ Hook to pass data structures into the class:
            takes the preloaded hashes that the KB
            uses to grab probability estimates.
        """
        self.KB_obj = queryKB(read_db=True, db=database)

    def get(self, halluc_set=[], preps=True, AllParses=False, MaxLikelihood=False, choose_PPs=True, form_data={}):
        # Arguments are params for sofie.
        self.sofie_obj = Sofie(self.KB_obj, halluc_set=halluc_set, with_preps=preps, choose_PPs=choose_PPs)
        self.print_head()
        # Arguments are output options for sofie.
        self.render_images(AllParses=AllParses, MaxLikelihood=MaxLikelihood)
        self.print_tail(form_data)

    def post(self):
        """ Figures out the parameters that have been set,
            sends these through to self.sofie_obj in get.
        """
        MaxLikelihood = False
        AllParses = False
        halluc_set = []
        form_data = {}
        chosen_string = self.get_argument("string")
        if chosen_string == "Yes":
            MaxLikelihood = True
            form_data["string"] = True
            choose_PPs = False
        else:
            form_data["string"] = False
            choose_PPs = True
        chosen_all = self.get_argument("all")
        if chosen_all == "Yes":
            AllParses = True
            form_data["all"] = True
        else:
            form_data["all"] = False
        chosen_verb = self.get_argument("verb")
        if chosen_verb == "Yes":
            halluc_set += ["verb"]
            form_data["verb"] = True
        else:
            form_data["verb"] = False
        chosen_prep = self.get_argument("prep")
        if chosen_prep == "Yes":
            preps = False
            form_data["prep"] = True
        else:
            preps = True
            form_data["prep"] = False
        #print "form_data is", form_data
        self.get(halluc_set, preps, AllParses, MaxLikelihood, choose_PPs, form_data)

    def print_head(self):
        open_head = open(bin_path + "head.html", "r")
        read_head = open_head.read()
        open_head.close()
        self.write(read_head)

    def print_tail(self, form_data):
        tail = open(bin_path + "tail.html", "r")
        read_tail = tail.read()
        tail.close()
        # There are so many better ways to do this.
        for option in form_data:
            #print "Looking at option", option, "which is", form_data[option]
            if form_data[option] == True:
                read_tail = re.sub(option + "\" value=\"Yes\">", option + "\" value=\"Yes\" checked>", read_tail)
                read_tail = re.sub(option + "\" value=\"No\" checked>", option + "\" value=\"No\">", read_tail)
        self.write(read_tail)

    def print_image_description(self, image, vision_out, all_parses, chosen_description, pretty_description):
        all_parses = re.sub("'", r"\\'", all_parses)
        pretty_description = re.sub("'", r"\\'", pretty_description)
        chosen_description = re.sub("'", r"\\'", chosen_description)
        vision_out = re.sub("\n(?!\-)", "<BR />&nbsp;&nbsp;", vision_out)
        vision_out = re.sub("\n", "<BR />", vision_out)
        vision_out = re.sub("'(\d+),(\d+)'", r"(\1, \2)", vision_out)
        vision_out = re.sub("'", r"\\'", vision_out)
        self.write("<IMG src=\"" + image_path + image + ".jpg\" height=\"100\" id=" + image + " value=" + image + " onclick=\"output('" + image_path + "', '" + image + "', '" + all_parses + "', '" + chosen_description + "', '" + pretty_description + "', '" + str(vision_out) + "')\" \>")

    def do_yaml(self, in_file):
        """ Processes yaml-style input from computer vision. """
        prep_hash = {}
        obj_in_img = []
        n = 0
        split_file = in_file.split("\n")
        for line in split_file:
            line = line.strip()
            if line == "":
                continue
            split_line = line.split()
            if split_line[0] == "-":
                if n > 0:
                    obj_in_img += [{'id':id, 'post_id':post_id, \
                                    'type':type, 'label':label, \
                                    'score':score, 'bbox':[], 'attrs':attr_hash}]
                id = str(split_line[-1])
            elif split_line[0] == "type:":
                type = int(split_line[-1])
            elif split_line[0] == "label:":
                label = split_line[-1]
            elif split_line[0] == "score:":
                score = float(split_line[-1])
            elif split_line[0] == "post_id:":
                post_id = split_line[-1]
            elif split_line[0] == "bbox:":
                pass
            elif split_line[0] == "attrs:":
                attr_hash = eval(" ".join(split_line[1:]))
            elif split_line[0] == "preps:":
                prep_hash = eval(" ".join(split_line[1:]))
            n += 1
        obj_in_img += [{'id':id, 'post_id':post_id, \
                        'type':type, 'label':label, \
                        'score':score, 'bbox':[], \
                        'attrs':attr_hash, 'preps':prep_hash}]
        return obj_in_img

    def get_words(self, sentence):
        """ Prettifies output string (removes parse-tree info) """
        words = re.findall("\([^\(\) ]+ ([^\(\) ]+) [^\(\) ]+\)", sentence)
        pretty_description = ""
        for word in words:
            if word == "-":
                if len(words) == 2:
                    word = "some"
                else:
                    continue
            pretty_description += word + " "
        return pretty_description[0].upper() + pretty_description[1:-1] + "."

    def get_score(self, sentence):
        vals = re.findall("\([^\(\) ]+ [^\(\) ]+ ([^\(\) ]+)\)", sentence)
        num_vals = float(len(vals))
        # (Negative) Cross Entropy calculation
        product = 0.0
        for val in vals:
            product += float(val)
        return product/num_vals

    def max_like(self, generated_descriptions):
        final_description = ""
        final_pretty_description = ""
        for sentence in generated_descriptions:
            max_score = -99999999
            chosen_description = ""
            for description in generated_descriptions[sentence]:
                score = self.get_score(description)
                if score > max_score:
                    chosen_description = description
                    max_score = score
            final_description += "<BR />" + chosen_description
            final_pretty_description += "<BR />" + self.get_words(chosen_description)    
        if final_description == "":
            final_description = "This is a picture."
        return (final_description, final_pretty_description)

    def get_longest(self, generated_descriptions):
        final_description = ""
        final_pretty_description = ""
        for sentence in generated_descriptions:
            longest = 0
            chosen_description = ""
            for description in generated_descriptions[sentence]:
                pretty_description = self.get_words(description)
                if len(pretty_description) > longest:
                    chosen_description = description
                    chosen_pretty_description = pretty_description
                    longest = len(pretty_description)
            final_description += "<BR />" + chosen_description
            final_pretty_description += "<BR />" + chosen_pretty_description
        if final_description == "":
            final_description = "This is a picture."
        return (final_description, final_pretty_description)

    def run_sofie(self, vision_out, image_id, MaxLikelihood=False, AllParses=False):
        obj_in_img = self.do_yaml(vision_out)
        self.sofie_obj.get_detections(obj_in_img)
        generated_descriptions = self.sofie_obj.run()
        all_parses = ""
        if MaxLikelihood is True:
            #print "doing xent"
            (chosen_description, pretty_description) = self.max_like(generated_descriptions[image_id + ".txt"])
        else:
            #print "doing longest"
            (chosen_description, pretty_description) = self.get_longest(generated_descriptions[image_id + ".txt"])
        if AllParses:
            for sentence in generated_descriptions[image_id + ".txt"]:
                for parse_string in generated_descriptions[image_id + ".txt"][sentence]:
                    all_parses += "<BR />" + parse_string
        return (all_parses, chosen_description, pretty_description)

    def render_images(self, MaxLikelihood=False, AllParses=False):
        imagefid = open(bin_path + "images-dev-short.txt","r")
        imagefile = imagefid.readlines()
        imagefid.close()
        n = 0
        for line in imagefile:
            line = line.strip()
            chosen_image = line
            vision_fid = open(vision_path + chosen_image + ".txt", "r")
            vision_out = str(vision_fid.read())
            (all_parses, chosen_description, pretty_description) = self.run_sofie(vision_out, chosen_image, MaxLikelihood, AllParses)
            self.print_image_description(chosen_image, vision_out, all_parses, chosen_description, pretty_description)
            n += 1
            #if n > 50:
            #    break

settings = {
    "static_path": os.path.join(os.path.dirname(__file__), "static")
}

database = KB_obj.get_as_db()
app = tornado.web.Application([(r"/", MainHandler, dict(database=database))], **settings)

if __name__=="__main__":
    app.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
