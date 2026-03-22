# ---------------------------------------------------------------------------
# Modified by nestivi
# Modifications: Added a Polish language lexicon, implemented dynamic 
# grammatical inflection, and adapted the logic to support double negation.
# Original code licensed under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------

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

# nouns = [
#     "aktor", "artysta", "kamerdyner", "oszust", "dyrektor",
#     "ekspert", "rybak", "sędzia", "przysięgły", "malarz", "muzyk",
#     "policjant", "strażak", "profesor", "szeryf", "żołnierz", "student",
#     "filozof", "nauczyciel", "turysta", "prawnik", "lekarz", "inżynier",
#     "weterynarz", "dentysta", "księgowy", "technik", "elektryk",
#     "psycholog", "fizyk", "hydraulik", "kelner", "mechanik", "kucharz",
#     "bibliotekarz", "fryzjer", "ekonomista", "barman", "kasjer", "chirurg",
#     "pilot", "rzeźnik", "optyk", "sportowiec", "sprzątacz",
#     "aktuariusz", "żeglarz", "terapeuta", "tajny_agent", "hodowca_zwierząt", "kontroler_ruchu_lotniczego",
#     "antropolog", "treser_zwierząt", "alergolog", "agent_nieruchomości", "archeolog",
#     "astronom", "trener_atletyczny", "audiolog", "audytor", "woźny_sądowy",
#     "piekarz", "fryzjer_męski", "urzędnik", "kartograf", "kręgarz", "tancerz", "epidemiolog",
#     "rolnik", "florysta", "leśniczy", "kierowca_ciężarówki", "jubiler", "projektant_wnętrz",
#     "maszynista", "matematyk", "sekretarz", "fotograf", "spiker_radiowy", "dekarz",
#     "brukarz", "taksówkarz", "historyk", "poeta", "kaskader", "monologista", "wydawca",
#     "skryba", "bloger", "redaktor", "prezes", "kontroler_biletów", "zawiadowca_stacji", "geodeta",
#     "wiertacz", "uczony", "analityk_ilościowy", "dyrektor_finansowy", "dyrektor_techniczny", "dyrektor_it", "informatyk", "więzień",
#     "gość", "odwiedzający", "pomocnik", "żywiciel", "gospodarz", "duch", "rozgrywający", "strzelec",
#     "osadnik", "zdobywca", "cynik", "wiedźma", "kapitan", "analityk_biznesowy", "naukowiec_danych",
#     "handlowiec", "dyrektor_szkoły", "baletnica", "piłkarz", "krykiecista", "tenisista", "wykładowca",
#     "pacjent", "naukowiec_ai", "rowerzysta", "szachista", "strateg",
#     "naukowiec", "rodzic", "agent_fbi", "obrońca", "napastnik", "watażka", "inżynier_nlp",
#     "arcymistrz", "mistrz", "król", "królowa", "rycerz", "książę", "księżniczka", "niemowlę", "dorosły",
#     "doradca", "zapaśnik", "wojownik", "bokser", "pszczelarz", "dj", "skrzypek",
#     "dyrygent", "gimnastyk"
# ]

# count_furniture = [
#     "krzesło", "stół", "biurko", "taboret", "kanapa", "regał",
#     "łóżko", "materac", "komoda", "futon", "stolik_nocny", "pojemnik_do_przechowywania",
#     "hamak", "stół_bilardowy", "pianino", "szachownica", "drzwi"
# ]

# count_animals = [
#     "mrówkojad_afrykański", "pies", "alpaka", "pancernik", "mrówkojad", "pingwin",
#     "mrówka", "niedźwiedź", "bonobo", "bóbr", "ptak", "sowa", "motyl",
#     "bawół", "trzmiel", "żaba", "wieloryb", "bizon", "borsuk", "pawian",
#     "nosorożec", "wielbłąd", "kot", "kurczak", "gepard", "kakadu", "krowa", "krab",
#     "gąsienica", "szympans", "nur", "pająk", "krokodyl", "kojot", "szynszyla",
#     "kaczka", "jeleń", "delfin", "dingo", "osioł", "węgorz", "słoń", "emu", "goryl", "sokół",
#     "lis", "fretka", "gerbil", "pasikonik", "suseł", "koza", "hiena", "koń", "hipopotam",
#     "jaguar", "kangur", "lemur", "lew", "ryś", "jaszczurka", "świstak", "norka", "piżmak", "mysz",
#     "ara", "łoś", "traszka", "struś", "wydra", "świnia", "maskonur", "puma", "pelikan", "paw",
#     "królik", "wąż", "renifer", "szop", "szczur", "owca", "sęp", "wombat", "wilk", "guziec",
#     "mors", "łasica", "dzik", "zebra", "foka"
# ]

# verbs = [
#     "lubić", "podziwiać", "robić", "psuć", "zatrudniać",
#     "uderzać", "zabijać", "walczyć", "dotykać", "zgładzić",
#     "aprobować", "bronić", "zastępować", "gonić", "polować",
#     "nie_lubić", "rozpoznawać", "rozumieć", "czuć",
#     "kochać", "nienawidzić", "imponować", "wiedzieć", "zauważać", "dostrzegać",
#     "widzieć", "pamiętać", "zaskakiwać", "woleć",
#     "rysować", "oskarżać", "uwielbiać", "doradzać", "doceniać",
#     "podchodzić", "zadziwiać", "potrzebować", "wołać", "wierzyć",
#     "naśladować", "służyć", "konsultować", "przekonywać", "krytykować",
#     "pragnąć", "wątpić", "zachęcać", "badać",
#     "karmić", "wybaczać", "przytulać", "prowadzić_dochodzenie", "całować",
#     "wspominać", "wisieć_dłużnym", "namawiać", "proponować", "obiecywać",
#     "uderzyć_pięścią", "strzelać", "grozić", "tolerować", "ostrzegać",
#     "szanować", "podziwiać_z_zachwytem", "fantazjować", "użytkować", "mordować",
#     "wspierać"
# ]

# =========================================================================
# SYSTEM LEKSYKALNY (SŁOWNIK)
lexicon = {word: {"M": word, "N": word, "B": word, "D": word} for word in nouns + count_furniture + count_animals}
verbs_lexicon = {verb: {"si": verb, "pl": verb} for verb in verbs}
lexicon.update({
    # === RZECZOWNIKI (OSOBY) ===
    "aktor": {"M": "aktor", "N": "aktorem", "B": "aktora", "D": "aktora"},
    "artysta": {"M": "artysta", "N": "artystą", "B": "artystę", "D": "artysty"},
    "kamerdyner": {"M": "kamerdyner", "N": "kamerdynerem", "B": "kamerdynera", "D": "kamerdynera"},
    "oszust": {"M": "oszust", "N": "oszustem", "B": "oszusta", "D": "oszusta"},
    "dyrektor": {"M": "dyrektor", "N": "dyrektorem", "B": "dyrektora", "D": "dyrektora"},
    "ekspert": {"M": "ekspert", "N": "ekspertem", "B": "eksperta", "D": "eksperta"},
    "rybak": {"M": "rybak", "N": "rybakiem", "B": "rybaka", "D": "rybaka"},
    "sędzia": {"M": "sędzia", "N": "sędzią", "B": "sędziego", "D": "sędziego"},
    "przysięgły": {"M": "przysięgły", "N": "przysięgłym", "B": "przysięgłego", "D": "przysięgłego"},
    "malarz": {"M": "malarz", "N": "malarzem", "B": "malarza", "D": "malarza"},
    "muzyk": {"M": "muzyk", "N": "muzykiem", "B": "muzyka", "D": "muzyka"},
    "policjant": {"M": "policjant", "N": "policjantem", "B": "policjanta", "D": "policjanta"},
    "strażak": {"M": "strażak", "N": "strażakiem", "B": "strażaka", "D": "strażaka"},
    "profesor": {"M": "profesor", "N": "profesorem", "B": "profesora", "D": "profesora"},
    "szeryf": {"M": "szeryf", "N": "szeryfem", "B": "szeryfa", "D": "szeryfa"},
    "żołnierz": {"M": "żołnierz", "N": "żołnierzem", "B": "żołnierza", "D": "żołnierza"},
    "student": {"M": "student", "N": "studentem", "B": "studenta", "D": "studenta"},
    "filozof": {"M": "filozof", "N": "filozofem", "B": "filozofa", "D": "filozofa"},
    "nauczyciel": {"M": "nauczyciel", "N": "nauczycielem", "B": "nauczyciela", "D": "nauczyciela"},
    "turysta": {"M": "turysta", "N": "turystą", "B": "turystę", "D": "turysty"},
    "prawnik": {"M": "prawnik", "N": "prawnikiem", "B": "prawnika", "D": "prawnika"},
    "lekarz": {"M": "lekarz", "N": "lekarzem", "B": "lekarza", "D": "lekarza"},
    "inżynier": {"M": "inżynier", "N": "inżynierem", "B": "inżyniera", "D": "inżyniera"},
    "weterynarz": {"M": "weterynarz", "N": "weterynarzem", "B": "weterynarza", "D": "weterynarza"},
    "dentysta": {"M": "dentysta", "N": "dentystą", "B": "dentystę", "D": "dentysty"},
    "księgowy": {"M": "księgowy", "N": "księgowym", "B": "księgowego", "D": "księgowego"},
    "technik": {"M": "technik", "N": "technikiem", "B": "technika", "D": "technika"},
    "elektryk": {"M": "elektryk", "N": "elektrykiem", "B": "elektryka", "D": "elektryka"},
    "psycholog": {"M": "psycholog", "N": "psychologiem", "B": "psychologa", "D": "psychologa"},
    "fizyk": {"M": "fizyk", "N": "fizykiem", "B": "fizyka", "D": "fizyka"},
    "hydraulik": {"M": "hydraulik", "N": "hydraulikiem", "B": "hydraulika", "D": "hydraulika"},
    "kelner": {"M": "kelner", "N": "kelnerem", "B": "kelnera", "D": "kelnera"},
    "mechanik": {"M": "mechanik", "N": "mechanikiem", "B": "mechanika", "D": "mechanika"},
    "kucharz": {"M": "kucharz", "N": "kucharzem", "B": "kucharza", "D": "kucharza"},
    "bibliotekarz": {"M": "bibliotekarz", "N": "bibliotekarzem", "B": "bibliotekarza", "D": "bibliotekarza"},
    "fryzjer": {"M": "fryzjer", "N": "fryzjerem", "B": "fryzjera", "D": "fryzjera"},
    "ekonomista": {"M": "ekonomista", "N": "ekonomistą", "B": "ekonomistę", "D": "ekonomisty"},
    "barman": {"M": "barman", "N": "barmanem", "B": "barmana", "D": "barmana"},
    "kasjer": {"M": "kasjer", "N": "kasjerem", "B": "kasjera", "D": "kasjera"},
    "chirurg": {"M": "chirurg", "N": "chirurgiem", "B": "chirurga", "D": "chirurga"},
    "pilot": {"M": "pilot", "N": "pilotem", "B": "pilota", "D": "pilota"},
    "rzeźnik": {"M": "rzeźnik", "N": "rzeźnikiem", "B": "rzeźnika", "D": "rzeźnika"},
    "optyk": {"M": "optyk", "N": "optykiem", "B": "optyka", "D": "optyka"},
    "sportowiec": {"M": "sportowiec", "N": "sportowcem", "B": "sportowca", "D": "sportowca"},
    "sprzątacz": {"M": "sprzątacz", "N": "sprzątaczem", "B": "sprzątacza", "D": "sprzątacza"},
    "aktuariusz": {"M": "aktuariusz", "N": "aktuariuszem", "B": "aktuariusza", "D": "aktuariusza"},
    "żeglarz": {"M": "żeglarz", "N": "żeglarzem", "B": "żeglarza", "D": "żeglarza"},
    "terapeuta": {"M": "terapeuta", "N": "terapeutą", "B": "terapeutę", "D": "terapeuty"},
    "tajny_agent": {"M": "tajny agent", "N": "tajnym agentem", "B": "tajnego agenta", "D": "tajnego agenta"},
    "hodowca_zwierząt": {"M": "hodowca zwierząt", "N": "hodowcą zwierząt", "B": "hodowcę zwierząt", "D": "hodowcy zwierząt"},
    "kontroler_ruchu_lotniczego": {"M": "kontroler ruchu lotniczego", "N": "kontrolerem ruchu lotniczego", "B": "kontrolera ruchu lotniczego", "D": "kontrolera ruchu lotniczego"},
    "antropolog": {"M": "antropolog", "N": "antropologiem", "B": "antropologa", "D": "antropologa"},
    "treser_zwierząt": {"M": "treser zwierząt", "N": "treserem zwierząt", "B": "tresera zwierząt", "D": "tresera zwierząt"},
    "alergolog": {"M": "alergolog", "N": "alergologiem", "B": "alergologa", "D": "alergologa"},
    "agent_nieruchomości": {"M": "agent nieruchomości", "N": "agentem nieruchomości", "B": "agenta nieruchomości", "D": "agenta nieruchomości"},
    "archeolog": {"M": "archeolog", "N": "archeologiem", "B": "archeologa", "D": "archeologa"},
    "astronom": {"M": "astronom", "N": "astronomem", "B": "astronoma", "D": "astronoma"},
    "trener_atletyczny": {"M": "trener atletyczny", "N": "trenerem atletycznym", "B": "trenera atletycznego", "D": "trenera atleticznego"},
    "audiolog": {"M": "audiolog", "N": "audiologiem", "B": "audiologa", "D": "audiologa"},
    "audytor": {"M": "audytor", "N": "audytorem", "B": "audytora", "D": "audytora"},
    "woźny_sądowy": {"M": "woźny sądowy", "N": "woźnym sądowym", "B": "woźnego sądowego", "D": "woźnego sądowego"},
    "piekarz": {"M": "piekarz", "N": "piekarzem", "B": "piekarza", "D": "piekarza"},
    "fryzjer_męski": {"M": "fryzjer męski", "N": "fryzjerem męskim", "B": "fryzjera męskiego", "D": "fryzjera męskiego"},
    "urzędnik": {"M": "urzędnik", "N": "urzędnikiem", "B": "urzędnika", "D": "urzędnika"},
    "kartograf": {"M": "kartograf", "N": "kartografem", "B": "kartografa", "D": "kartografa"},
    "kręgarz": {"M": "kręgarz", "N": "kręgarzem", "B": "kręgarza", "D": "kręgarza"},
    "tancerz": {"M": "tancerz", "N": "tancerzem", "B": "tancerza", "D": "tancerza"},
    "epidemiolog": {"M": "epidemiolog", "N": "epidemiologiem", "B": "epidemiologa", "D": "epidemiologa"},
    "rolnik": {"M": "rolnik", "N": "rolnikiem", "B": "rolnika", "D": "rolnika"},
    "florysta": {"M": "florysta", "N": "florystą", "B": "florystę", "D": "florysty"},
    "leśniczy": {"M": "leśniczy", "N": "leśniczym", "B": "leśniczego", "D": "leśniczego"},
    "kierowca_ciężarówki": {"M": "kierowca ciężarówki", "N": "kierowcą ciężarówki", "B": "kierowcę ciężarówki", "D": "kierowcy ciężarówki"},
    "jubiler": {"M": "jubiler", "N": "jubilerem", "B": "jubilera", "D": "jubilera"},
    "projektant_wnętrz": {"M": "projektant wnętrz", "N": "projektantem wnętrz", "B": "projektanta wnętrz", "D": "projektanta wnętrz"},
    "maszynista": {"M": "maszynista", "N": "maszynistą", "B": "maszynistę", "D": "maszynisty"},
    "matematyk": {"M": "matematyk", "N": "matematykiem", "B": "matematyka", "D": "matematyka"},
    "sekretarz": {"M": "sekretarz", "N": "sekretarzem", "B": "sekretarza", "D": "sekretarza"},
    "fotograf": {"M": "fotograf", "N": "fotografem", "B": "fotografa", "D": "fotografa"},
    "spiker_radiowy": {"M": "spiker radiowy", "N": "spikerem radiowym", "B": "spikera radiowego", "D": "spikera radiowego"},
    "dekarz": {"M": "dekarz", "N": "dekarzem", "B": "dekarza", "D": "dekarza"},
    "brukarz": {"M": "brukarz", "N": "brukarzem", "B": "brukarza", "D": "brukarza"},
    "taksówkarz": {"M": "taksówkarz", "N": "taksówkarzem", "B": "taksówkarza", "D": "taksówkarza"},
    "historyk": {"M": "historyk", "N": "historykiem", "B": "historyka", "D": "historyka"},
    "poeta": {"M": "poeta", "N": "poetą", "B": "poetę", "D": "poety"},
    "kaskader": {"M": "kaskader", "N": "kaskaderem", "B": "kaskadera", "D": "kaskadera"},
    "monologista": {"M": "monologista", "N": "monologistą", "B": "monologistę", "D": "monologisty"},
    "wydawca": {"M": "wydawca", "N": "wydawcą", "B": "wydawcę", "D": "wydawcy"},
    "skryba": {"M": "skryba", "N": "skrybą", "B": "skrybę", "D": "skryby"},
    "bloger": {"M": "bloger", "N": "blogerem", "B": "blogera", "D": "blogera"},
    "redaktor": {"M": "redaktor", "N": "redaktorem", "B": "redaktora", "D": "redaktora"},
    "prezes": {"M": "prezes", "N": "prezesem", "B": "prezesa", "D": "prezesa"},
    "kontroler_biletów": {"M": "kontroler biletów", "N": "kontrolerem biletów", "B": "kontrolera biletów", "D": "kontrolera biletów"},
    "zawiadowca_stacji": {"M": "zawiadowca stacji", "N": "zawiadowcą stacji", "B": "zawiadowcę stacji", "D": "zawiadowcy stacji"},
    "geodeta": {"M": "geodeta", "N": "geodetą", "B": "geodetę", "D": "geodety"},
    "wiertacz": {"M": "wiertacz", "N": "wiertaczem", "B": "wiertacza", "D": "wiertacza"},
    "uczony": {"M": "uczony", "N": "uczonym", "B": "uczonego", "D": "uczonego"},
    "analityk_ilościowy": {"M": "analityk ilościowy", "N": "analitykiem ilościowym", "B": "analityka ilościowego", "D": "analityka ilościowego"},
    "dyrektor_finansowy": {"M": "dyrektor finansowy", "N": "dyrektorem finansowym", "B": "dyrektora finansowego", "D": "dyrektora finansowego"},
    "dyrektor_techniczny": {"M": "dyrektor techniczny", "N": "dyrektorem technicznym", "B": "dyrektora technicznego", "D": "dyrektora technicznego"},
    "dyrektor_it": {"M": "dyrektor IT", "N": "dyrektorem IT", "B": "dyrektora IT", "D": "dyrektora IT"},
    "informatyk": {"M": "informatyk", "N": "informatykiem", "B": "informatyka", "D": "informatyka"},
    "więzień": {"M": "więzień", "N": "więźniem", "B": "więźnia", "D": "więźnia"},
    "gość": {"M": "gość", "N": "gościem", "B": "gościa", "D": "gościa"},
    "odwiedzający": {"M": "odwiedzający", "N": "odwiedzającym", "B": "odwiedzającego", "D": "odwiedzającego"},
    "pomocnik": {"M": "pomocnik", "N": "pomocnikiem", "B": "pomocnika", "D": "pomocnika"},
    "żywiciel": {"M": "żywiciel", "N": "żywicielem", "B": "żywiciela", "D": "żywiciela"},
    "gospodarz": {"M": "gospodarz", "N": "gospodarzem", "B": "gospodarza", "D": "gospodarza"},
    "duch": {"M": "duch", "N": "duchem", "B": "ducha", "D": "ducha"},
    "rozgrywający": {"M": "rozgrywający", "N": "rozgrywającym", "B": "rozgrywającego", "D": "rozgrywającego"},
    "strzelec": {"M": "strzelec", "N": "strzelcem", "B": "strzelca", "D": "strzelca"},
    "osadnik": {"M": "osadnik", "N": "osadnikiem", "B": "osadnika", "D": "osadnika"},
    "zdobywca": {"M": "zdobywca", "N": "zdobywcą", "B": "zdobywcę", "D": "zdobywcy"},
    "cynik": {"M": "cynik", "N": "cynikiem", "B": "cynika", "D": "cynika"},
    "wiedźma": {"M": "wiedźma", "N": "wiedźmą", "B": "wiedźmę", "D": "wiedźmy"},
    "kapitan": {"M": "kapitan", "N": "kapitanem", "B": "kapitana", "D": "kapitana"},
    "analityk_biznesowy": {"M": "analityk biznesowy", "N": "analitykiem biznesowym", "B": "analityka biznesowego", "D": "analityka biznesowego"},
    "naukowiec_danych": {"M": "naukowiec danych", "N": "naukowcem danych", "B": "naukowca danych", "D": "naukowca danych"},
    "handlowiec": {"M": "handlowiec", "N": "handlowcem", "B": "handlowca", "D": "handlowca"},
    "dyrektor_szkoły": {"M": "dyrektor szkoły", "N": "dyrektorem szkoły", "B": "dyrektora szkoły", "D": "dyrektora szkoły"},
    "baletnica": {"M": "baletnica", "N": "baletnicą", "B": "baletnicę", "D": "baletnicy"},
    "piłkarz": {"M": "piłkarz", "N": "piłkarzem", "B": "piłkarza", "D": "piłkarza"},
    "krykiecista": {"M": "krykiecista", "N": "krykiecistą", "B": "krykiecistę", "D": "krykiecisty"},
    "tenisista": {"M": "tenisista", "N": "tenisistą", "B": "tenisistę", "D": "tenisisty"},
    "wykładowca": {"M": "wykładowca", "N": "wykładowcą", "B": "wykładowcę", "D": "wykładowcy"},
    "pacjent": {"M": "pacjent", "N": "pacjentem", "B": "pacjenta", "D": "pacjenta"},
    "naukowiec_ai": {"M": "naukowiec AI", "N": "naukowcem AI", "B": "naukowca AI", "D": "naukowca AI"},
    "rowerzysta": {"M": "rowerzysta", "N": "rowerzystą", "B": "rowerzystę", "D": "rowerzysty"},
    "szachista": {"M": "szachista", "N": "szachistą", "B": "szachistę", "D": "szachisty"},
    "strateg": {"M": "strateg", "N": "strategiem", "B": "stratega", "D": "stratega"},
    "naukowiec": {"M": "naukowiec", "N": "naukowcem", "B": "naukowca", "D": "naukowca"},
    "rodzic": {"M": "rodzic", "N": "rodzicem", "B": "rodzica", "D": "rodzica"},
    "agent_fbi": {"M": "agent FBI", "N": "agentem FBI", "B": "agenta FBI", "D": "agenta FBI"},
    "obrońca": {"M": "obrońca", "N": "obrońcą", "B": "obrońcę", "D": "obrońcy"},
    "napastnik": {"M": "napastnik", "N": "napastnikiem", "B": "napastnika", "D": "napastnika"},
    "watażka": {"M": "watażka", "N": "watażką", "B": "watażkę", "D": "watażki"},
    "inżynier_nlp": {"M": "inżynier NLP", "N": "inżynierem NLP", "B": "inżyniera NLP", "D": "inżyniera NLP"},
    "arcymistrz": {"M": "arcymistrz", "N": "arcymistrzem", "B": "arcymistrza", "D": "arcymistrza"},
    "mistrz": {"M": "mistrz", "N": "mistrzem", "B": "mistrza", "D": "mistrza"},
    "król": {"M": "król", "N": "królem", "B": "króla", "D": "króla"},
    "królowa": {"M": "królowa", "N": "królową", "B": "królową", "D": "królowej"},
    "rycerz": {"M": "rycerz", "N": "rycerzem", "B": "rycerza", "D": "rycerza"},
    "książę": {"M": "książę", "N": "księciem", "B": "księcia", "D": "księcia"},
    "księżniczka": {"M": "księżniczka", "N": "księżniczką", "B": "księżniczkę", "D": "księżniczki"},
    "niemowlę": {"M": "niemowlę", "N": "niemowlęciem", "B": "niemowlę", "D": "niemowlęcia"},
    "dorosły": {"M": "dorosły", "N": "dorosłym", "B": "dorosłego", "D": "dorosłego"},
    "doradca": {"M": "doradca", "N": "doradcą", "B": "doradcę", "D": "doradcy"},
    "zapaśnik": {"M": "zapaśnik", "N": "zapaśnikiem", "B": "zapaśnika", "D": "zapaśnika"},
    "wojownik": {"M": "wojownik", "N": "wojownikiem", "B": "wojownika", "D": "wojownika"},
    "bokser": {"M": "bokser", "N": "bokserem", "B": "boksera", "D": "boksera"},
    "pszczelarz": {"M": "pszczelarz", "N": "pszczelarzem", "B": "pszczelarza", "D": "pszczelarza"},
    "dj": {"M": "DJ", "N": "DJ-em", "B": "DJ-a", "D": "DJ-a"},
    "skrzypek": {"M": "skrzypek", "N": "skrzypkiem", "B": "skrzypka", "D": "skrzypka"},
    "dyrygent": {"M": "dyrygent", "N": "dyrygentem", "B": "dyrygenta", "D": "dyrygenta"},
    "gimnastyk": {"M": "gimnastyk", "N": "gimnastykiem", "B": "gimnastyka", "D": "gimnastyka"},

    # === MEBLE I PRZEDMIOTY (Rzeczowniki nieżywotne - Biernik = Mianownik) ===
    "krzesło": {"M": "krzesło", "N": "krzesłem", "B": "krzesło", "D": "krzesła"},
    "stół": {"M": "stół", "N": "stołem", "B": "stół", "D": "stołu"},
    "biurko": {"M": "biurko", "N": "biurkiem", "B": "biurko", "D": "biurka"},
    "taboret": {"M": "taboret", "N": "taboretem", "B": "taboret", "D": "taboretu"},
    "kanapa": {"M": "kanapa", "N": "kanapą", "B": "kanapę", "D": "kanapy"},
    "regał": {"M": "regał", "N": "regałem", "B": "regał", "D": "regału"},
    "łóżko": {"M": "łóżko", "N": "łóżkiem", "B": "łóżko", "D": "łóżka"},
    "materac": {"M": "materac", "N": "materacem", "B": "materac", "D": "materaca"},
    "komoda": {"M": "komoda", "N": "komodą", "B": "komodę", "D": "komody"},
    "futon": {"M": "futon", "N": "futonem", "B": "futon", "D": "futonu"},
    "stolik_nocny": {"M": "stolik nocny", "N": "stolikiem nocnym", "B": "stolik nocny", "D": "stolika nocnego"},
    "pojemnik_do_przechowywania": {"M": "pojemnik do przechowywania", "N": "pojemnikiem do przechowywania", "B": "pojemnik do przechowywania", "D": "pojemnika do przechowywania"},
    "hamak": {"M": "hamak", "N": "hamakiem", "B": "hamak", "D": "hamaka"},
    "stół_bilardowy": {"M": "stół bilardowy", "N": "stołem bilardowym", "B": "stół bilardowy", "D": "stołu bilardowego"},
    "pianino": {"M": "pianino", "N": "pianinem", "B": "pianino", "D": "pianina"},
    "szachownica": {"M": "szachownica", "N": "szachownicą", "B": "szachownicę", "D": "szachownicy"},
    "drzwi": {"M": "drzwi", "N": "drzwiami", "B": "drzwi", "D": "drzwi"},

    # === ZWIERZĘTA ===
    "mrówkojad_afrykański": {"M": "mrówkojad afrykański", "N": "mrówkojadem afrykańskim", "B": "mrówkojada afrykańskiego", "D": "mrówkojada afrykańskiego"},
    "pies": {"M": "pies", "N": "psem", "B": "psa", "D": "psa"},
    "alpaka": {"M": "alpaka", "N": "alpaką", "B": "alpakę", "D": "alpaki"},
    "pancernik": {"M": "pancernik", "N": "pancernikiem", "B": "pancernika", "D": "pancernika"},
    "mrówkojad": {"M": "mrówkojad", "N": "mrówkojadem", "B": "mrówkojada", "D": "mrówkojada"},
    "pingwin": {"M": "pingwin", "N": "pingwinem", "B": "pingwina", "D": "pingwina"},
    "mrówka": {"M": "mrówka", "N": "mrówką", "B": "mrówkę", "D": "mrówki"},
    "niedźwiedź": {"M": "niedźwiedź", "N": "niedźwiedziem", "B": "niedźwiedzia", "D": "niedźwiedzia"},
    "bonobo": {"M": "bonobo", "N": "bonobo", "B": "bonobo", "D": "bonobo"},
    "bóbr": {"M": "bóbr", "N": "bobrem", "B": "bobra", "D": "bobra"},
    "ptak": {"M": "ptak", "N": "ptakiem", "B": "ptaka", "D": "ptaka"},
    "sowa": {"M": "sowa", "N": "sową", "B": "sowę", "D": "sowy"},
    "motyl": {"M": "motyl", "N": "motylem", "B": "motyla", "D": "motyla"},
    "bawół": {"M": "bawół", "N": "bawołem", "B": "bawoła", "D": "bawoła"},
    "trzmiel": {"M": "trzmiel", "N": "trzmielem", "B": "trzmiela", "D": "trzmiela"},
    "żaba": {"M": "żaba", "N": "żabą", "B": "żabę", "D": "żaby"},
    "wieloryb": {"M": "wieloryb", "N": "wielorybem", "B": "wieloryba", "D": "wieloryba"},
    "bizon": {"M": "bizon", "N": "bizonem", "B": "bizona", "D": "bizona"},
    "borsuk": {"M": "borsuk", "N": "borsukiem", "B": "borsuka", "D": "borsuka"},
    "pawian": {"M": "pawian", "N": "pawianem", "B": "pawiana", "D": "pawiana"},
    "nosorożec": {"M": "nosorożcem", "N": "nosorożcem", "B": "nosorożca", "D": "nosorożca"},
    "wielbłąd": {"M": "wielbłąd", "N": "wielbłądem", "B": "wielbłąda", "D": "wielbłąda"},
    "kot": {"M": "kot", "N": "kotem", "B": "kota", "D": "kota"},
    "kurczak": {"M": "kurczak", "N": "kurczakiem", "B": "kurczaka", "D": "kurczaka"},
    "gepard": {"M": "gepard", "N": "gepardem", "B": "geparda", "D": "geparda"},
    "kakadu": {"M": "kakadu", "N": "kakadu", "B": "kakadu", "D": "kakadu"},
    "krowa": {"M": "krowa", "N": "krową", "B": "krowę", "D": "krowy"},
    "krab": {"M": "krab", "N": "krabem", "B": "kraba", "D": "kraba"},
    "gąsienica": {"M": "gąsienica", "N": "gąsienicą", "B": "gąsienicę", "D": "gąsienicy"},
    "szympans": {"M": "szympans", "N": "szympansem", "B": "szympansa", "D": "szympansa"},
    "nur": {"M": "nur", "N": "nurem", "B": "nura", "D": "nura"},
    "pająk": {"M": "pająk", "N": "pająkiem", "B": "pająka", "D": "pająka"},
    "krokodyl": {"M": "krokodyl", "N": "krokodylem", "B": "krokodyla", "D": "krokodyla"},
    "kojot": {"M": "kojot", "N": "kojotem", "B": "kojota", "D": "kojota"},
    "szynszyla": {"M": "szynszyla", "N": "szynszylą", "B": "szynszylę", "D": "szynszyli"},
    "kaczka": {"M": "kaczka", "N": "kaczką", "B": "kaczkę", "D": "kaczki"},
    "jeleń": {"M": "jeleń", "N": "jeleniem", "B": "jelenia", "D": "jelenia"},
    "delfin": {"M": "delfin", "N": "delfinem", "B": "delfina", "D": "delfina"},
    "dingo": {"M": "dingo", "N": "dingo", "B": "dingo", "D": "dingo"},
    "osioł": {"M": "osioł", "N": "osłem", "B": "osła", "D": "osła"},
    "węgorz": {"M": "węgorz", "N": "węgorzem", "B": "węgorza", "D": "węgorza"},
    "słoń": {"M": "słoń", "N": "słoniem", "B": "słonia", "D": "słonia"},
    "emu": {"M": "emu", "N": "emu", "B": "emu", "D": "emu"},
    "goryl": {"M": "goryl", "N": "gorylem", "B": "goryla", "D": "goryla"},
    "sokół": {"M": "sokół", "N": "sokołem", "B": "sokoła", "D": "sokoła"},
    "lis": {"M": "lis", "N": "lisem", "B": "lisa", "D": "lisa"},
    "fretka": {"M": "fretka", "N": "fretką", "B": "fretkę", "D": "fretki"},
    "gerbil": {"M": "gerbil", "N": "gerbilem", "B": "gerbila", "D": "gerbila"},
    "pasikonik": {"M": "pasikonik", "N": "pasikonikiem", "B": "pasikonika", "D": "pasikonika"},
    "suseł": {"M": "suseł", "N": "susłem", "B": "susła", "D": "susła"},
    "koza": {"M": "koza", "N": "kozą", "B": "kozę", "D": "kozy"},
    "hiena": {"M": "hiena", "N": "hieną", "B": "hienę", "D": "hieny"},
    "koń": {"M": "koń", "N": "koniem", "B": "konia", "D": "konia"},
    "hipopotam": {"M": "hipopotam", "N": "hipopotamem", "B": "hipopotama", "D": "hipopotama"},
    "jaguar": {"M": "jaguar", "N": "jaguarem", "B": "jaguara", "D": "jaguara"},
    "kangur": {"M": "kangur", "N": "kangurem", "B": "kangura", "D": "kangura"},
    "lemur": {"M": "lemur", "N": "lemurem", "B": "lemura", "D": "lemura"},
    "lew": {"M": "lew", "N": "lwem", "B": "lwa", "D": "lwa"},
    "ryś": {"M": "ryś", "N": "rysiem", "B": "rysia", "D": "rysia"},
    "jaszczurka": {"M": "jaszczurka", "N": "jaszczurką", "B": "jaszczurkę", "D": "jaszczurki"},
    "świstak": {"M": "świstak", "N": "świstakiem", "B": "świstaka", "D": "świstaka"},
    "norka": {"M": "norka", "N": "norką", "B": "norkę", "D": "norki"},
    "piżmak": {"M": "piżmak", "N": "piżmakiem", "B": "piżmaka", "D": "piżmaka"},
    "mysz": {"M": "mysz", "N": "myszą", "B": "mysz", "D": "myszy"},
    "ara": {"M": "ara", "N": "arą", "B": "arę", "D": "ary"},
    "łoś": {"M": "łoś", "N": "łosiem", "B": "łosia", "D": "łosia"},
    "traszka": {"M": "traszka", "N": "traszką", "B": "traszkę", "D": "traszki"},
    "struś": {"M": "struś", "N": "strusiem", "B": "strusia", "D": "strusia"},
    "wydra": {"M": "wydra", "N": "wydrą", "B": "wydrę", "D": "wydry"},
    "świnia": {"M": "świnia", "N": "świnią", "B": "świnię", "D": "świni"},
    "maskonur": {"M": "maskonur", "N": "maskonurem", "B": "maskonura", "D": "maskonura"},
    "puma": {"M": "puma", "N": "pumą", "B": "pumę", "D": "pumy"},
    "pelikan": {"M": "pelikan", "N": "pelikanem", "B": "pelikana", "D": "pelikana"},
    "paw": {"M": "paw", "N": "pawiem", "B": "pawia", "D": "pawia"},
    "królik": {"M": "królik", "N": "królikiem", "B": "królika", "D": "królika"},
    "wąż": {"M": "wąż", "N": "wężem", "B": "węża", "D": "węża"},
    "renifer": {"M": "renifer", "N": "reniferem", "B": "renifera", "D": "renifera"},
    "szop": {"M": "szop", "N": "szopem", "B": "szopa", "D": "szopa"},
    "szczur": {"M": "szczur", "N": "szczurem", "B": "szczura", "D": "szczura"},
    "owca": {"M": "owca", "N": "owcą", "B": "owcę", "D": "owcy"},
    "sęp": {"M": "sęp", "N": "sępem", "B": "sępa", "D": "sępa"},
    "wombat": {"M": "wombat", "N": "wombatem", "B": "wombata", "D": "wombata"},
    "wilk": {"M": "wilk", "N": "wilkiem", "B": "wilka", "D": "wilka"},
    "guziec": {"M": "guziec", "N": "guźcem", "B": "guźca", "D": "guźca"},
    "mors": {"M": "mors", "N": "morsem", "B": "morsa", "D": "morsa"},
    "łasica": {"M": "łasica", "N": "łasicą", "B": "łasicę", "D": "łasicy"},
    "dzik": {"M": "dzik", "N": "dzikiem", "B": "dzika", "D": "dzika"},
    "zebra": {"M": "zebra", "N": "zebrą", "B": "zebrę", "D": "zebry"},
    "foka": {"M": "foka", "N": "foką", "B": "fokę", "D": "foki"},

    # === CZASOWNIKI ===
    "lubić": {"si": "lubi", "pl": "lubią"},
    "podziwiać": {"si": "podziwia", "pl": "podziwiają"},
    "robić": {"si": "robi", "pl": "robią"},
    "psuć": {"si": "psuje", "pl": "psują"},
    "zatrudniać": {"si": "zatrudnia", "pl": "zatrudniają"},
    "uderzać": {"si": "uderza", "pl": "uderzają"},
    "zabijać": {"si": "zabija", "pl": "zabijają"},
    "walczyć": {"si": "walczy", "pl": "walczą"},
    "dotykać": {"si": "dotyka", "pl": "dotykają"},
    "zgładzić": {"si": "zgładzi", "pl": "zgładzą"},
    "aprobować": {"si": "aprobuje", "pl": "aprobują"},
    "bronić": {"si": "broni", "pl": "bronią"},
    "zastępować": {"si": "zastępuje", "pl": "zastępują"},
    "gonić": {"si": "goni", "pl": "gonią"},
    "polować": {"si": "poluje", "pl": "polują"},
    "nie_lubić": {"si": "nie lubi", "pl": "nie lubią"},
    "rozpoznawać": {"si": "rozpoznaje", "pl": "rozpoznają"},
    "rozumieć": {"si": "rozumie", "pl": "rozumieją"},
    "czuć": {"si": "czuje", "pl": "czują"},
    "kochać": {"si": "kocha", "pl": "kochają"},
    "nienawidzić": {"si": "nienawidzi", "pl": "nienawidzą"},
    "imponować": {"si": "imponuje", "pl": "imponują"},
    "wiedzieć": {"si": "wie", "pl": "wiedzą"},
    "zauważać": {"si": "zauważa", "pl": "zauważają"},
    "dostrzegać": {"si": "dostrzega", "pl": "dostrzegają"},
    "widzieć": {"si": "widzi", "pl": "widzą"},
    "pamiętać": {"si": "pamięta", "pl": "pamiętają"},
    "zaskakiwać": {"si": "zaskakuje", "pl": "zaskakują"},
    "woleć": {"si": "woli", "pl": "wolą"},
    "rysować": {"si": "rysuje", "pl": "rysują"},
    "oskarżać": {"si": "oskarża", "pl": "oskarżają"},
    "uwielbiać": {"si": "uwielbia", "pl": "uwielbiają"},
    "doradzać": {"si": "doradza", "pl": "doradzają"},
    "doceniać": {"si": "docenia", "pl": "doceniają"},
    "podchodzić": {"si": "podchodzi", "pl": "podchodzą"},
    "zadziwiać": {"si": "zadziwia", "pl": "zadziwiają"},
    "potrzebować": {"si": "potrzebuje", "pl": "potrzebują"},
    "wołać": {"si": "woła", "pl": "wołają"},
    "wierzyć": {"si": "wierzy", "pl": "wierzą"},
    "naśladować": {"si": "naśladuje", "pl": "naśladują"},
    "służyć": {"si": "służy", "pl": "służą"},
    "konsultować": {"si": "konsultuje", "pl": "konsultują"},
    "przekonywać": {"si": "przekonuje", "pl": "przekonują"},
    "krytykować": {"si": "krytykuje", "pl": "krytykują"},
    "pragnąć": {"si": "pragnie", "pl": "pragną"},
    "wątpić": {"si": "wątpi", "pl": "wątpią"},
    "zachęcać": {"si": "zachęca", "pl": "zachęcają"},
    "badać": {"si": "bada", "pl": "badają"},
    "karmić": {"si": "karmi", "pl": "karmią"},
    "wybaczać": {"si": "wybacza", "pl": "wybaczają"},
    "przytulać": {"si": "przytula", "pl": "przytulają"},
    "prowadzić_dochodzenie": {"si": "prowadzi dochodzenie", "pl": "prowadzą dochodzenie"},
    "całować": {"si": "całuje", "pl": "całują"},
    "wspominać": {"si": "wspomina", "pl": "wspominają"},
    "wisieć_dłużnym": {"si": "wisi dłużnym", "pl": "wiszą dłużnym"},
    "namawiać": {"si": "namawia", "pl": "namawiają"},
    "proponować": {"si": "proponuje", "pl": "proponują"},
    "obiecywać": {"si": "obiecuje", "pl": "obiecują"},
    "uderzyć_pięścią": {"si": "uderza pięścią", "pl": "uderzają pięścią"},
    "strzelać": {"si": "strzela", "pl": "strzelają"},
    "grozić": {"si": "grozi", "pl": "grożą"},
    "tolerować": {"si": "toleruje", "pl": "tolerują"},
    "ostrzegać": {"si": "ostrzega", "pl": "ostrzegają"},
    "szanować": {"si": "szanuje", "pl": "szanują"},
    "podziwiać_z_zachwytem": {"si": "podziwia z zachwytem", "pl": "podziwiają z zachwytem"},
    "fantazjować": {"si": "fantazjuje", "pl": "fantazjują"},
    "użytkować": {"si": "użytkuje", "pl": "użytkują"},
    "mordować": {"si": "morduje", "pl": "mordują"},
    "wspierać": {"si": "wspiera", "pl": "wspierają"}
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