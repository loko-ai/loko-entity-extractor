import re
from abc import ABC, abstractmethod
from typing import List

import dateparser
import phonenumbers
from codicefiscale import codicefiscale
from schwifty import IBAN, BIC
from vat_validator import validate_vat

MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
          "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
          "january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]

MONTHS_ABBR = ["gen", "feb", "mar", "apr", "mag", "giu",
               "lug", "ago", "set", "ott", "nov", "dic",
               "jan", "may", "jun", "jul", "aug", "sep", "oct", "dec"]


class BaseEntityExtractor(ABC):

    def __call__(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise ValueError("Invalid input parameter 'text'!")
        return self.validate(self.extract(" " + text + " "))

    def __str__(self):
        return str(self.__class__.__name__)

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def extract(self, text: str) -> List[str]:
        pass

    def validate(self, extracted_items: List[str]) -> List[str]:
        return extracted_items


class CFExtractor(BaseEntityExtractor):

    def __init__(self):
        self.field = "CF"
        self.reg = "[a-zA-Z]{3}\s?[a-zA-Z]{3}\s?[0-9]{2}[a-zA-Z]{1}[0-9]{2}\s?[a-zA-Z]{1}[0-9]{3}[a-zA-Z]{1}"

    def extract(self, text: str) -> List[str]:
        return re.findall(self.reg, text)

    def validate(self, extracted_items: List[str]) -> List[str]:
        codici_fiscali = set()
        for prob_CF in extracted_items:
            prob_CF = prob_CF.strip()
            if codicefiscale.is_valid(prob_CF):
                codici_fiscali.add(prob_CF)
        return list(codici_fiscali)


class DateExtractor(BaseEntityExtractor):

    def __init__(self):
        self.field = "DATES"
        self.reg1 = "(\d+\s*[/\-\.]\s*\d+\s*[/\-\.]\s*\d+)"

    def extract(self, text: str) -> List[str]:
        l1 = re.findall(self.reg1, text)
        for m in MONTHS_ABBR + MONTHS + [x.capitalize() for x in MONTHS_ABBR] + [x.capitalize() for x in MONTHS]:
            reg2 = "(\d*\s+" + m + "\s+\d+)"
            l2 = re.findall(reg2, text)
            reg3 = "(\d+\s*[/\-\.]\s*" + m + "\s*[/\-\.]\s*\d+)"
            l3 = re.findall(reg3, text)
            l1 += l2
            l1 += l3
        result = set()
        for date_str in l1:
            date_str = date_str.strip()
            result.add(date_str)
        return list(result)

    def validate(self, extracted_items: List[str]) -> List[str]:
        result = []
        for date_str in extracted_items:
            date = dateparser.parse(date_str, languages=['it'],
                                    settings={'DATE_ORDER': 'DMY', 'PREFER_DAY_OF_MONTH': 'first'})
            if date is not None:
                # date_str_formatted = date.date().strftime("%d/%m/%Y")
                result.append(date_str)
        return result


class PartitaIVAextractor(BaseEntityExtractor):

    def __init__(self):
        self.field = "VAT_NUMBERS"

    def extract(self, text: str) -> List[str]:
        partite_iva = set()
        for token in text.split():
            t = token.strip()
            if t.isdigit() and 6 <= len(t) <= 11:
                it_t = "IT" + str(t.zfill(11))
            elif len(t) == 13 and (t.startswith("IT") or t.startswith("it")) and str(t[2:]).isdigit():
                it_t = t
            else:
                continue
            if validate_vat('IT', it_t):
                partite_iva.add(t)
        return list(partite_iva)


class TargaExtractor(BaseEntityExtractor):

    def __init__(self):
        self.field = "PLATE_NUMBERS"
        self.ita_auto_reg = "[\s,;.:!?\*\('\"\[\{][A-Z]{2}\s*[0-9]{3}\s*[A-Z]{2}[\s,;.:!?\*\)'\"\]\}]"  # autoveicoli
        self.ita_moto_reg = "[\s,;.:!?\*\('\"\[\{][A-Z]{2}\s*[0-9]{3}\s*[0-9]{2}[\s,;.:!?\*\)'\"\]\}]"  # motoveicoli
        self.ita_ciclo_reg = "[\s,;.:!?\*\('\"\[\{][BCDFGHJKLMNOPRSTVWXYZ23456789]{6}[\s,;.:!?\*\)'\"\]\}]"  # ciclomotori

    def extract(self, text: str) -> List[str]:
        l1 = re.findall(self.ita_auto_reg, text)
        l2 = re.findall(self.ita_moto_reg, text)
        l3 = re.findall(self.ita_ciclo_reg, text)
        result = set()
        for targa in l1 + l2 + l3:
            targa = targa.strip(",;.:!?()*'\"[]{}").strip()
            result.add(targa)
        return list(result)


class EmailExtractor(BaseEntityExtractor):

    def __init__(self):
        self.field = "EMAILS"
        self.reg1 = "[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    def extract(self, text: str) -> List[str]:
        result = set()
        l1 = re.findall(self.reg1, text)
        for em in l1:
            em = em.strip(".").strip()
            result.add(em)
        return list(result)


class TelephoneNumberExtractor(BaseEntityExtractor):

    def __init__(self):
        self.field = "TEL_NUMBERS"

    def extract(self, text: str) -> List[str]:
        result = set()
        for match in phonenumbers.PhoneNumberMatcher(text, "IT"):
            tn = match.raw_string.strip(",;.:!?()*'\"[]{}").strip()
            result.add(tn)
        return list(result)


class InternationalBankAccountNumberExtractor(BaseEntityExtractor):

    def __init__(self):
        self.field = "IBAN"
        # self.ita_regex ="IT\s*[0-9]{2}\s*[a-zA-Z]{1}\s*[0-9]{5}\s*[0-9]{5}\s*[a-zA-Z0-9]{12}"
        self.ita_regex = "IT\s*[0-9]{2}\s*[a-zA-Z]{1}\s*[0-9\s]{22,100}"

    def extract(self, text: str) -> List[str]:
        result = set()
        for token in text.split():
            t = token.strip()
            if 15 <= len(t) <= 34 and str(t[0:2]).isalpha() and str(t[2:4]).isdigit() and str(t[4:]).isalnum():
                result.add(t)
        l1 = re.findall(self.ita_regex, text)  # search italian iban with spaces inside
        for ita_iban in l1:
            result.add(ita_iban.strip(",;.:!?()*").strip())
        return list(result)

    def validate(self, extracted_items: List[str]) -> List[str]:
        result = []
        for item in extracted_items:
            try:
                if len(item.split()) > 1:
                    ita_iban2 = "".join(item.split())
                    iban = IBAN(ita_iban2)
                else:
                    iban = IBAN(item)
                if iban.validate():  # if IBAN is valid
                    result.append(item)
            except Exception:  # if IBAN is not valid
                continue
        return result


class BankIdentifierCodeExtractor(BaseEntityExtractor):

    def __init__(self):
        self.field = "BIC"
        self.bic_reg1 = "[\s,;.:!?\*\('\"\[\{][A-Z]{6}[A-Z0-9]{2}[\s,;.:!?\*\)'\"\]\}]"
        self.bic_reg2 = "[\s,;.:!?\*\('\"\[\{][A-Z]{6}[A-Z0-9]{5}[\s,;.:!?\*\)'\"\]\}]"

    def extract(self, text: str) -> List[str]:
        result = set()
        l1 = re.findall(self.bic_reg1, text)
        l2 = re.findall(self.bic_reg1, text)
        for b in l1 + l2:
            b = b.strip(",;.:!?()*'\"[]{}").strip()
            result.add(b)
        return list(result)

    def validate(self, extracted_items: List[str]) -> List[str]:
        result = []
        for item in extracted_items:
            try:
                bic = BIC(item)
                if bic.validate():
                    result.append(item)
            except Exception:  # if BIC is not valid
                continue
        return result


class CurrencyExtractor(BaseEntityExtractor):

    def __init__(self):
        self.field = "CURRENCY"
        self.reg1 = "[0-9]+[,\.]*[0-9]*\s*[₤£$€]"
        self.reg2 = "[0-9]+[,\.]*[0-9]*\s*euro"
        self.reg3 = "[0-9]+[,\.]*[0-9]*\s*EUR"

    def extract(self, text: str) -> List[str]:
        l1 = re.findall(self.reg1, text)
        l2 = re.findall(self.reg2, text)
        l3 = re.findall(self.reg3, text)
        result = set()
        for currency in l1 + l2 + l3:
            cur = currency.strip()
            result.add(cur)
        return list(result)
