from z3 import *
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

import argparse

from nltk import *
from nltk.sem.drt import DrtParser
from nltk.sem import logic
import nltk
from nltk.sem import Expression
from nltk import load_parser
from nltk.sem import Valuation, Model
from nltk.corpus import brown

from scipy.stats import beta
from numpy import histogram

from typing import List, Dict, Any, Union, Optional
import io

import random
import re
import time

import pandas as pd

from nltk.parse.generate import generate
from fragments import (
  SyllogisticTemplates,
  RelationalSyllogiticTemplates, 
  RelativeClausesTemplates, 
  RelativeTVTemplates, 
  AnaphoraTemplates,
)

from tqdm import tqdm

nouns = [
    "aktor", "artysta", "kamerdyner", "oszust", "dyrektor",
    "ekspert", "rybak", "sędzia", "przysięgły", "malarz", "muzyk",
    "policjant", "strażak", "profesor", "szeryf", "żołnierz", "student",
    "filozof", "nauczyciel", "turysta", "prawnik", "lekarz", "inżynier",
    "weterynarz", "dentysta", "księgowy", "technik", "elektryk",
    "psycholog", "fizyk", "hydraulik", "kelner", "mechanik", "kucharz",
    "bibliotekarz", "fryzjer", "ekonomista", "barman", "kasjer", "chirurg",
    "pilot", "rzeźnik", "optyk", "sportowiec", "sprzątacz",
    "aktuariusz", "żeglarz", "terapeuta", "tajny_agent", "hodowca_zwierząt", "kontroler_ruchu_lotniczego",
    "antropolog", "treser_zwierząt", "alergolog", "agent_nieruchomości", "archeolog",
    "astronom", "trener_atletyczny", "audiolog", "audytor", "woźny_sądowy",
    "piekarz", "fryzjer_męski", "urzędnik", "kartograf", "kręgarz", "tancerz", "epidemiolog",
    "rolnik", "florysta", "leśniczy", "kierowca_ciężarówki", "jubiler", "projektant_wnętrz",
    "maszynista", "matematyk", "sekretarz", "fotograf", "spiker_radiowy", "dekarz",
    "brukarz", "taksówkarz", "historyk", "poeta", "kaskader", "monologista", "wydawca",
    "skryba", "bloger", "redaktor", "prezes", "kontroler_biletów", "zawiadowca_stacji", "geodeta",
    "wiertacz", "uczony", "analityk_ilościowy", "dyrektor_finansowy", "dyrektor_techniczny", "dyrektor_it", "informatyk", "więzień",
    "gość", "odwiedzający", "pomocnik", "żywiciel", "gospodarz", "duch", "rozgrywający", "strzelec",
    "osadnik", "zdobywca", "cynik", "wiedźma", "kapitan", "analityk_biznesowy", "naukowiec_danych",
    "handlowiec", "dyrektor_szkoły", "baletnica", "piłkarz", "krykiecista", "tenisista", "wykładowca",
    "pacjent", "naukowiec_ai", "rowerzysta", "szachista", "strateg",
    "naukowiec", "rodzic", "agent_fbi", "obrońca", "napastnik", "watażka", "inżynier_nlp",
    "arcymistrz", "mistrz", "król", "królowa", "rycerz", "książę", "księżniczka", "niemowlę", "dorosły",
    "doradca", "zapaśnik", "wojownik", "bokser", "pszczelarz", "dj", "skrzypek",
    "dyrygent", "gimnastyk"
]

count_furniture = [
    "krzesło", "stół", "biurko", "taboret", "kanapa", "regał",
    "łóżko", "materac", "komoda", "futon", "stolik_nocny", "pojemnik_do_przechowywania",
    "hamak", "stół_bilardowy", "pianino", "szachownica", "drzwi"
]

count_animals = [
    "mrówkojad_afrykański", "pies", "alpaka", "pancernik", "mrówkojad", "pingwin",
    "mrówka", "niedźwiedź", "bonobo", "bóbr", "ptak", "sowa", "motyl",
    "bawół", "trzmiel", "żaba", "wieloryb", "bizon", "borsuk", "pawian",
    "nosorożec", "wielbłąd", "kot", "kurczak", "gepard", "kakadu", "krowa", "krab",
    "gąsienica", "szympans", "nur", "pająk", "krokodyl", "kojot", "szynszyla",
    "kaczka", "jeleń", "delfin", "dingo", "osioł", "węgorz", "słoń", "emu", "goryl", "sokół",
    "lis", "fretka", "gerbil", "pasikonik", "suseł", "koza", "hiena", "koń", "hipopotam",
    "jaguar", "kangur", "lemur", "lew", "ryś", "jaszczurka", "świstak", "norka", "piżmak", "mysz",
    "ara", "łoś", "traszka", "struś", "wydra", "świnia", "maskonur", "puma", "pelikan", "paw",
    "królik", "wąż", "renifer", "szop", "szczur", "owca", "sęp", "wombat", "wilk", "guziec",
    "mors", "łasica", "dzik", "zebra", "foka"
]

verbs = [
    "lubić", "podziwiać", "robić", "psuć", "zatrudniać",
    "uderzać", "zabijać", "walczyć", "dotykać", "zgładzić",
    "aprobować", "bronić", "zastępować", "gonić", "polować",
    "nie_lubić", "rozpoznawać", "rozumieć", "czuć",
    "kochać", "nienawidzić", "imponować", "wiedzieć", "zauważać", "dostrzegać",
    "widzieć", "pamiętać", "zaskakiwać", "woleć",
    "rysować", "oskarżać", "uwielbiać", "doradzać", "doceniać",
    "podchodzić", "zadziwiać", "potrzebować", "wołać", "wierzyć",
    "naśladować", "służyć", "konsultować", "przekonywać", "krytykować",
    "pragnąć", "wątpić", "zachęcać", "badać",
    "karmić", "wybaczać", "przytulać", "prowadzić_dochodzenie", "całować",
    "wspominać", "wisieć_dłużnym", "namawiać", "proponować", "obiecywać",
    "uderzyć_pięścią", "strzelać", "grozić", "tolerować", "ostrzegać",
    "szanować", "podziwiać_z_zachwytem", "fantazjować", "użytkować", "mordować",
    "wspierać"
]

# =========================================================================
# SYSTEM LEKSYKALNY (SŁOWNIK)
# =========================================================================

# 1. Automatyczne wypełnienie słownika formami mianownika (zapobiega błędom KeyError)
lexicon = {word: {"M": word, "N": word, "B": word, "D": word} for word in nouns + count_furniture + count_animals}
verbs_lexicon = {verb: {"si": verb, "pl": verb} for verb in verbs}
lexicon.update(verbs_lexicon)

# 2. Ręczne nadpisywanie poprawnymi formami polskimi.
# TODO: Docelowo musisz tutaj wpisać odmianę dla wszystkich swoich słów!
lexicon.update({
    # RZECZOWNIKI (M: Mianownik, N: Narzędnik, B: Biernik, D: Dopełniacz)
    "aktor": {"M": "aktor", "N": "aktorem", "B": "aktora", "D": "aktora"},
    "artysta": {"M": "artysta", "N": "artystą", "B": "artystę", "D": "artysty"},
    "kamerdyner": {"M": "kamerdyner", "N": "kamerdynerem", "B": "kamerdynera", "D": "kamerdynera"},
    "oszust": {"M": "oszust", "N": "oszustem", "B": "oszusta", "D": "oszusta"},
    "dyrektor": {"M": "dyrektor", "N": "dyrektorem", "B": "dyrektora", "D": "dyrektora"},
    "strażak": {"M": "strażak", "N": "strażakiem", "B": "strażaka", "D": "strażaka"},
    "optyk": {"M": "optyk", "N": "optykiem", "B": "optyka", "D": "optyka"},
    
    # CZASOWNIKI (si: 3 os. l. pojedynczej, pl: 3 os. l. mnogiej)
    "lubić": {"si": "lubi", "pl": "lubią"},
    "podziwiać": {"si": "podziwia", "pl": "podziwiają"},
    "robić": {"si": "robi", "pl": "robią"},
    "nie_lubić": {"si": "nie lubi", "pl": "nie lubią"},
})
# =========================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="data construction SAT")

    parser.add_argument(
        "--fragment",
        type=str,
        default=None,
        required = True,
        help="The name of the language fragment to generate the data",
    )

    parser.add_argument(
        "--sampling_file",
        type=str,
        default=None,
        required = True,
        help="File contianing the disribution of satisfiablity of the language fragment",
    )

    parser.add_argument(
        "--min_ab",
        type=float,
        default=0.35,
        help="minimum prob value of the phase change region",
    )

    parser.add_argument(
        "--max_ab",
        type=float,
        default=0.65,
        help="maximum prob value of the phase change region",
    )

    parser.add_argument(
        "--max_a",
        type=int,
        default=5,
        help="maximum number of unary predicates",
    )

    parser.add_argument(
        "--max_b",
        type=int,
        default=5,
        help="maximum number of binary predicates",
    )

    parser.add_argument(
        "--min_a",
        type=int,
        default=3,
        help="minimum number of unary predicates",
    )

    parser.add_argument(
        "--min_b",
        type=int,
        default=2,
        help="minimum number of binary predicates",
    )

    parser.add_argument(
        "--time_out",
        type=int,
        default=10000,
        help="Timeout value for the Z3 theorem prover",
    )

    parser.add_argument(
        "--prob",
        type=float,
        default=0.5,
        help="probability of sentences belong to the more complex sub fragments whithin a datapoint",
    )

    parser.add_argument(
        "--num_datapoints",
        type=int,
        default=50000,
        help="number of datapoints",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default='fragment.csv',
    )

    args = parser.parse_args()

    return args


class LangaugeFragmentSAT:

  def __init__(self, functions, lexicon, language_fragment, df_hard, min_a = 3, max_a = 8, min_b = 3, max_b = 8, timeout = 10000, prob = 0.5, a_b = 2):

    # PRZEKAZANIE LEKSYKONU DO SZABLONÓW
    self.syl_templates = SyllogisticTemplates(functions, lexicon)
    self.relsyl_templates = RelationalSyllogiticTemplates(functions, lexicon)
    self.relative_templates = RelativeClausesTemplates(functions, lexicon)
    self.relative_tv_templates = RelativeTVTemplates(functions, lexicon)
    self.anaphora_templates = AnaphoraTemplates(functions, lexicon)
    
    self.langauge_fragment = language_fragment
    self.timeout = timeout
    self.df_hard = df_hard
    self.min_m_a = df_hard['m/a'].min()
    self.max_m_a = df_hard['m/a'].max()
    self.m_a = df_hard['m/a'].to_list()
    self.min_a = min_a
    self.max_a = max_a
    self.min_b = min_b
    self.max_b = max_b
    self.prob = prob
    self.dist = beta(a_b, a_b)

  def generate_syllogistic(self, nouns, verbs, x, y, unary = 3, binary = 3, num_clauses = 6):
    s = Solver()

    list_fol = []
    list_sentences = []
    list_quantifiers = []

    unary_preds = random.sample(nouns, unary)
    binary_preds = random.sample(verbs, binary)
    prob = 1

    for i in range(num_clauses):
      logic, sentence, quantifiers  = self.syl_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y, negations = True)
    
      list_fol.append(logic)
      list_sentences.append(sentence)
      list_quantifiers.append(quantifiers)

    s.add(list_fol)
    s.set("timeout", self.timeout)
    sat = str(s.check())

    return list_fol, list_sentences, list_quantifiers, sat, prob

  def generate_relative_clauses(self, nouns, verbs, x, y, unary = 3, binary = 3, num_clauses = 6):

    s = Solver()

    list_fol = []
    list_sentences = []
    list_quantifiers = []

    unary_preds = random.sample(nouns, unary)
    binary_preds = random.sample(verbs, binary)

    prob = self.prob

    for i in range(num_clauses):
      if random.uniform(0,1) < prob :
        logic, sentence, quantifiers  = self.relative_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
      else :
        logic, sentence, quantifiers  = self.syl_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)

      list_fol.append(logic)
      list_sentences.append(sentence)
      list_quantifiers.append(quantifiers)

    s.add(list_fol)

    s.set("timeout", self.timeout)
    sat = str(s.check())

    return list_fol, list_sentences, list_quantifiers, sat, prob

  def generate_relational_syllogistic(self, nouns, verbs, x, y, unary = 3, binary = 3, num_clauses = 6):

    s = Solver()

    list_fol = []
    list_sentences = []
    list_quantifiers = []

    unary_preds = random.sample(nouns, unary)
    binary_preds = random.sample(verbs, binary)

    prob = self.prob

    for i in range(num_clauses):
      if random.uniform(0,1) < prob :
        logic, sentence, quantifiers  = self.relsyl_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
      else :
        random_fragment = random.choice(["syl", "rel"])
        if random_fragment == "syl":
          logic, sentence, quantifiers  = self.syl_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
        else :
          logic, sentence, quantifiers  = self.relative_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
        
      list_fol.append(logic)
      list_sentences.append(sentence)
      list_quantifiers.append(quantifiers)

    s.add(list_fol)

    s.set("timeout", self.timeout)
    sat = str(s.check())

    return list_fol, list_sentences, list_quantifiers, sat, prob

  def generate_relative_tv(self, nouns, verbs, x, y, unary = 3, binary = 3, num_clauses = 6):

    s = Solver()

    list_fol = []
    list_sentences = []
    list_quantifiers = []

    unary_preds = random.sample(nouns, unary)
    binary_preds = random.sample(verbs, binary)

    prob = self.prob
    for i in range(num_clauses):
      if random.uniform(0,1) < prob :
        logic, sentence, quantifiers  = self.relative_tv_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
      else :
        random_fragment = random.choice(["syl", "re-syl", "rel"])
        if random_fragment == "syl":
          logic, sentence, quantifiers  = self.syl_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
        elif random_fragment == "re-syl" :
          logic, sentence, quantifiers  = self.relsyl_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
        else :
          logic, sentence, quantifiers  = self.relative_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
      list_fol.append(logic)
      list_sentences.append(sentence)
      list_quantifiers.append(quantifiers)

    s.add(list_fol)
    s.set("timeout", self.timeout)
    sat = str(s.check())

    return list_fol, list_sentences, list_quantifiers, sat, prob

  def generate_anaphora(self, nouns, verbs, x, y, unary = 3, binary = 3, num_clauses = 6):

    s = Solver()

    list_fol = []
    list_sentences = []
    list_quantifiers = []

    unary_preds = random.sample(nouns, unary)
    binary_preds = random.sample(verbs, binary)

    prob = self.prob
    for i in range(num_clauses):
      if random.uniform(0,1) < prob :
        logic, sentence, quantifiers  = self.anaphora_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
      else :
        random_fragment = random.choice(["syl", "re-syl", "rel","syl", "re-syl", "rel", "rel_tv"])
        if random_fragment == "syl":
          logic, sentence, quantifiers  = self.syl_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
        elif random_fragment == "re-syl" :
          logic, sentence, quantifiers  = self.relsyl_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
        elif random_fragment == "rel" :
          logic, sentence, quantifiers  = self.relative_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
        else :
          logic, sentence, quantifiers  = self.relative_tv_templates.generate_sentence_logic_pair(unary_preds, binary_preds, x, y)
      list_fol.append(logic)
      list_sentences.append(sentence)
      list_quantifiers.append(quantifiers)

    s.add(list_fol)
    s.set("timeout", self.timeout)
    sat = str(s.check())

    return list_fol, list_sentences, list_quantifiers, sat, prob

  def generate_datapoint(self, nouns, verbs, x, y, unary = 3, binary = 3, num_clauses = 6):
    if self.langauge_fragment == "syllogistic":
      time.sleep(0.01)
      return self.generate_syllogistic(nouns, verbs, x, y, unary, binary, num_clauses)
    elif self.langauge_fragment == "syllogistic minus":
      time.sleep(0.01)
      return self.generate_syllogistic_minus(nouns, verbs, x, y, unary, binary, num_clauses)
    elif self.langauge_fragment == "relational syllogistic":
      return self.generate_relational_syllogistic(nouns, verbs, x, y, unary, binary, num_clauses)
    elif self.langauge_fragment == "relative clauses":
      return self.generate_relative_clauses(nouns, verbs, x, y, unary, binary, num_clauses)
    elif self.langauge_fragment == "relative transitive verbs":
      time.sleep(0.01)
      return self.generate_relative_tv(nouns, verbs, x, y, unary, binary, num_clauses)
    elif self.langauge_fragment == "anaphora":
      time.sleep(0.01)
      return self.generate_anaphora(nouns, verbs, x, y, unary, binary, num_clauses)

  def generator(self):
    while True:
      yield

  def create_df(self, nouns, verbs, x, y, num_datapoints=10000):

    data = {
        "formulae" : [],
        "sentences" : [],
        "quantifiers" : [],
        "sat" : [],
        "unary" : [],
        "binary" : [],
        "num_clauses" : [],
        "prob" : []
    }
    
    count = 0
    iter = 0

    progress_bar = tqdm(range(num_datapoints))

    while (True):

      if (self.langauge_fragment == "syllogistic") or (self.langauge_fragment == "relative clauses") or ((self.langauge_fragment == "syllogistic minus")):

        sample = self.min_a + self.dist.rvs(size=1) * (self.max_a - self.min_a)

        unary = round(sample[0])
        num_clauses = round(random.uniform(self.min_m_a, self.max_m_a) * unary)
        binary = 1
      else :
        unary = random.randint(self.min_a, self.max_a)
        num_clauses = round(random.uniform(self.min_m_a, self.max_m_a) * unary)
        m_a_ratio = min(self.m_a, key=lambda x:abs(x - num_clauses/unary))

        min_m_b = self.df_hard[self.df_hard['m/a'] == m_a_ratio]['m/b'].min()
        max_m_b = self.df_hard[self.df_hard['m/a'] == m_a_ratio]['m/b'].max()
        
        binary = round(num_clauses / random.uniform(min_m_b, max_m_b))
        if (binary > self.max_b) or (binary < self.min_b) :
          continue 
      list_fol, list_sentences, list_quantifiers, sat, prob = self.generate_datapoint(nouns, verbs, x, y, unary, binary, num_clauses)

      iter += 1

      if sat != "unknown":
        data["formulae"].append(list_fol)
        data["sentences"].append(list_sentences)
        data["quantifiers"].append(list_quantifiers)
        data["sat"].append(sat)
        data["unary"].append(unary)
        data["binary"].append(binary)
        data["num_clauses"].append(num_clauses)
        data["prob"].append(prob)

        count += 1

        progress_bar.update(1)
                  
      if count >= num_datapoints :
        break

      if iter % 1000 == 0 :
        print(iter, count)

    df = pd.DataFrame(data)
    return df


def main():

    args = parse_args()

    set_param(proof=True)

    ctx = Context()

    Z = IntSort()
    B = BoolSort()

    x, y = Ints('x y')

    functions = {}
    for f in nouns :
        functions[f] = Function(f, Z, B)

    for f in verbs :
        functions[f] = Function(f, Z, Z, B)

    df_agg = pd.read_csv(args.sampling_file)
    df_hard = df_agg[(df_agg['is_sat'] < args.max_ab) & (df_agg['is_sat'] > args.min_ab)]

    # PRZEKAZANIE LEKSYKONU DO KLASY ZARZĄDZAJĄCEJ
    satFragment = LangaugeFragmentSAT(functions,
                                       lexicon,  # <--- ZMIANA
                                       args.fragment, 
                                       df_hard, 
                                       min_a = args.min_a, 
                                       max_a = args.max_a, 
                                       min_b = args.min_b, 
                                       max_b = args.max_b, 
                                       timeout = args.time_out, 
                                       prob = args.prob)
    df = satFragment.create_df(nouns, verbs, x, y, num_datapoints=args.num_datapoints)

    df.to_csv(args.output_file, index = False)

if __name__ == "__main__":
    main()