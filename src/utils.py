from __future__ import annotations

import re
import unicodedata

COUNTRY_TO_ISO = {
    "England": "GB-ENG", "Scotland": "GB-SCT", "Wales": "GB-WLS", "Northern Ireland": "GB-NIR",
    "Argentina": "AR", "Brazil": "BR", "France": "FR", "Germany": "DE", "Spain": "ES", "Portugal": "PT",
    "Netherlands": "NL", "Belgium": "BE", "Italy": "IT", "Croatia": "HR", "Uruguay": "UY", "Mexico": "MX",
    "United States": "US", "USA": "US", "Canada": "CA", "South Korea": "KR", "Korea Republic": "KR",
    "Japan": "JP", "Morocco": "MA", "Ghana": "GH", "Senegal": "SN", "Switzerland": "CH", "Czech Republic": "CZ",
    "Czechia": "CZ", "South Africa": "ZA", "Australia": "AU", "Turkey": "TR", "Egypt": "EG", "Colombia": "CO",
    "Norway": "NO", "Sweden": "SE", "Denmark": "DK", "Poland": "PL", "Austria": "AT", "Serbia": "RS",
    "Ecuador": "EC", "Paraguay": "PY", "Iran": "IR", "New Zealand": "NZ", "Saudi Arabia": "SA", "Tunisia": "TN",
    "Ivory Coast": "CI", "Côte d'Ivoire": "CI", "DR Congo": "CD", "Algeria": "DZ", "Qatar": "QA",
    "Bosnia and Herzegovina": "BA", "Cape Verde": "CV", "Haiti": "HT", "Panama": "PA", "Iraq": "IQ",
    "Jordan": "JO", "Uzbekistan": "UZ", "Curaçao": "CW", "Curacao": "CW",
}

ABBR_TO_COUNTRY = {
    "ENG":"England", "SCO":"Scotland", "WAL":"Wales", "ARG":"Argentina", "BRA":"Brazil", "FRA":"France",
    "GER":"Germany", "ESP":"Spain", "POR":"Portugal", "NED":"Netherlands", "BEL":"Belgium", "ITA":"Italy",
    "CRO":"Croatia", "URU":"Uruguay", "MEX":"Mexico", "USA":"United States", "CAN":"Canada", "KOR":"South Korea",
    "JPN":"Japan", "MAR":"Morocco", "GHA":"Ghana", "SEN":"Senegal", "SUI":"Switzerland", "CZE":"Czechia",
    "RSA":"South Africa", "AUS":"Australia", "TUR":"Turkey", "EGY":"Egypt", "COL":"Colombia", "NOR":"Norway",
    "SWE":"Sweden", "AUT":"Austria", "ECU":"Ecuador", "PAR":"Paraguay", "IRI":"Iran", "NZL":"New Zealand",
    "KSA":"Saudi Arabia", "TUN":"Tunisia", "CIV":"Ivory Coast", "COD":"DR Congo", "DZA":"Algeria", "QAT":"Qatar",
    "BIH":"Bosnia and Herzegovina", "CPV":"Cape Verde", "HTI":"Haiti", "PAN":"Panama", "IRQ":"Iraq", "JOR":"Jordan",
    "UZB":"Uzbekistan", "CUW":"Curacao"
}

def normalize_name(name: str) -> str:
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def iso_to_flag(iso: str) -> str:
    if not iso or not isinstance(iso, str):
        return ""
    if iso.startswith("GB-"):
        return {"GB-ENG": "🏴", "GB-SCT": "🏴", "GB-WLS": "🏴", "GB-NIR": "🇬🇧"}.get(iso, "🇬🇧")
    iso = iso.upper()[:2]
    if len(iso) != 2 or not iso.isalpha():
        return ""
    return chr(ord(iso[0]) + 127397) + chr(ord(iso[1]) + 127397)


def country_to_flag(country: str) -> str:
    return iso_to_flag(COUNTRY_TO_ISO.get(str(country), ""))
