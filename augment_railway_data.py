import argparse
import itertools
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET = 10000
RNG = random.Random(42)

# Curated base rules focused on realistic Indian Railways policy language.
CANONICAL_RULES = [
    "a passenger travelling without a valid ticket must pay excess fare and penalty as per railway rules",
    "a passenger travelling beyond the booked destination must pay difference in fare and applicable charges",
    "a passenger travelling in a higher class than booked must pay fare difference and excess charges",
    "waiting list passengers are not allowed to board reserved sleeper or ac coaches",
    "waiting list passengers may travel in unreserved general coaches only",
    "rac passengers are allowed to travel and may receive a full berth after chart preparation",
    "tatkal tickets are generally non-refundable except in cases notified by indian railways",
    "if a train is cancelled by railways passengers are eligible for a full refund",
    "a tdr claim should be filed within the prescribed time on irctc or prs channels",
    "duplicate reserved ticket may be issued by railway authorities after payment of the prescribed fee",
    "passengers must carry a valid identity proof while travelling on an e-ticket",
    "name correction or transfer on a reserved ticket is allowed only as per railway guidelines",
    "journey extension from a reserved station should be informed to the ticket checking staff",
    "pnr status can be checked through irctc website ntes app or railway enquiry channels",
    "railway complaints may be filed on rail madad portal or through helpline 139",
    "overcharging by onboard catering staff can be reported to railway complaint channels",
    "smoking in train coaches is prohibited and attracts penalty",
    "carrying inflammable explosive or hazardous material in coaches is prohibited",
    "a person creating nuisance in intoxicated condition in railway premises is liable for action",
    "a person obstructing a railway employee on duty is punishable under railway law",
    "unauthorized chain pulling without valid reason is punishable",
    "trespassing on railway tracks is prohibited and punishable",
    "travelling on footboard rooftop or between coaches is prohibited for safety reasons",
    "male passengers are not allowed in compartments reserved for women",
    "passengers should not carry luggage that blocks passage or emergency exits",
    "railway staff may remove passengers who refuse to show ticket during checking",
    "ticketless travel cases are handled by ticket checking staff as per railway act",
    "platform ticket is required for non-travelling visitors entering paid areas",
    "reserved coach entry is restricted to passengers with valid reservation",
    "berth allotment is done by reservation system and final chart rules",
    "lower berth preference is provided to senior citizens as per quota rules",
    "lower berth preference is provided to women aged forty five years and above",
    "lower berth preference is provided to pregnant women subject to berth availability",
    "berth preference selected during booking is not a guaranteed allotment",
    "persons with disabilities are eligible for railway concessions as per notified rules",
    "wheelchair or assistance services can be requested at major stations",
    "medical help can be requested through onboard staff or railway helpline",
    "if accommodation is not provided despite confirmed reservation passenger may claim refund as per rules",
    "passengers should arrive early at stations to complete security and boarding formalities",
    "excess baggage should be booked in luggage office when weight exceeds free allowance",
    "baggage limit in second class non-ac is generally thirty five kilograms per passenger",
    "baggage limit in sleeper class is generally forty kilograms per passenger",
    "baggage limit in ac classes is generally fifty kilograms per passenger",
    "high-value baggage should be declared and booked as per railway luggage rules",
    "animals are permitted for travel only under prescribed railway conditions",
    "pet dogs may be booked in first ac coupe as per rules and charges",
    "e-ticket cancellation charges depend on class and time before departure",
    "refund on missed journey depends on tdr filing and railway claim policy",
    "group booking and concession booking are subject to document verification",
    "student concession is applicable only where officially permitted by railways",
    "patient concessions are available only for notified medical categories",
    "defence and other concessional fares apply only with valid authorization",
    "season ticket holders must travel within validity period and permitted class",
    "mobile ticket or sms confirmation should be produced during ticket inspection",
    "final reservation chart is prepared before departure as per railway schedule",
    "berth upgrades may happen automatically based on availability and quota release",
    "no refund is granted for lost unreserved tickets except where specifically allowed",
    "reserved ticket cancellation through irctc follows online refund timelines",
    "offline counter ticket refunds are processed through prs counters as per rules",
    "train delay compensation rules apply only in specific notified situations",
    "a confirmed passenger denied boarding due to overbooking may seek remedy as per railway rules",
    "passengers are advised to verify coach position and platform details before boarding",
    "luggage with suspicious content may be checked by security authorities",
    "unclaimed luggage found in trains or stations is handled under lost property procedures",
    "children below prescribed age may travel without separate berth under railway policy",
    "children requiring separate berth must have valid ticket",
    "passengers should board only at the station specified in booking unless rules permit otherwise",
    "break journey on long distance tickets is allowed only under prescribed conditions",
    "transfer of ticket to family member is allowed only at designated counters and within deadline",
    "railway can reschedule or regulate trains for operational safety and maintenance",
    "special trains may have separate fare and refund rules notified by railway administration",
    "premium tatkal tickets follow dynamic fare and dedicated cancellation policy",
    "passengers must cooperate with ticket checking and security verification during journey",
    "verbal abuse or assault on railway staff invites legal action",
    "cleanliness and waste disposal rules in coaches must be followed by passengers",
    "charging high power appliances in coaches is not permitted unless authorized",
    "unauthorized vending or solicitation inside coaches is prohibited",
    "station announcements and official apps should be treated as primary travel information sources",
    "emergency medical chain pulling is allowed only for genuine and immediate safety reasons",
    "coach and berth numbers should be cross-checked with final chart before boarding",
    "refund for partially travelled journey is governed by excess fare and tdr provisions",
    "railway administration may deny travel to passengers with contagious disease in public interest",
]

PREFIXES = [
    "",
    "as per indian railways rules ",
    "according to railway policy ",
    "railway guideline states that ",
    "under current railway norms ",
    "in indian railways operations ",
    "official railway instructions mention that ",
    "for passenger guidance railway rules say ",
    "as per reservation manual ",
    "as per ticketing policy ",
    "railway administration clarifies that ",
    "for train travel compliance ",
    "under railway passenger charter ",
    "as per official railway advisory ",
    "under standard operating railway rules ",
]

SUFFIXES = [
    "",
    " as per notified railway policy",
    " under railway act provisions",
    " as per applicable circulars",
    " subject to operational conditions",
    " unless revised by competent authority",
    " according to reservation and ticketing manual",
    " as per commercial department instructions",
    " as per coaching tariff norms",
    " as per current railway notification",
    " under passenger amenity standards",
    " as per onboard ticket checking protocol",
    " under standard customer service procedures",
]

SUBSTITUTIONS = {
    "not allowed": ["prohibited", "not permitted"],
    "penalty": ["fine", "penal action"],
    "refund": ["fare refund", "reimbursement"],
    "ticket checking staff": ["tte", "travelling ticket examiner"],
    "journey": ["travel"],
    "travelling": ["traveling"],
    "authorities": ["officials"],
}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lexical_variants(rule: str) -> set[str]:
    variants = {rule}
    for src, targets in SUBSTITUTIONS.items():
        if src in rule:
            for target in targets:
                pattern = rf"\\b{re.escape(src)}\\b"
                variants.add(re.sub(pattern, target, rule))

    # Combine up to 2 substitutions for stronger diversity.
    keys = [k for k in SUBSTITUTIONS if k in rule]
    for a, b in itertools.combinations(keys, 2):
        for ta in SUBSTITUTIONS[a]:
                for tb in SUBSTITUTIONS[b]:
                    step = re.sub(rf"\\b{re.escape(a)}\\b", ta, rule)
                    variants.add(re.sub(rf"\\b{re.escape(b)}\\b", tb, step))

    return {normalize(v) for v in variants if v.strip()}


def build_dataset(target: int) -> list[str]:
    bases = [normalize(r) for r in CANONICAL_RULES]
    pool = set()

    for base in bases:
        for variant in lexical_variants(base):
            for prefix in PREFIXES:
                for suffix in SUFFIXES:
                    candidate = normalize(f"{prefix}{variant}{suffix}")
                    if len(candidate.split()) >= 9:
                        pool.add(candidate)

    if len(pool) < target:
        raise RuntimeError(f"Could only generate {len(pool)} rules, lower than target {target}.")

    ordered = sorted(pool)
    RNG.shuffle(ordered)
    return ordered[:target]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic synthetic railway rule corpus")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "railway_data.txt",
    )
    args = parser.parse_args()

    target = max(args.target, 10000)
    lines = build_dataset(target)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Generated lines: {len(lines)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
