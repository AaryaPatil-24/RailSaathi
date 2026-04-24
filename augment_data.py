import random
from itertools import product

def create_augmented_rules():
    seen = set()
    rules = []
    
    templates = [
        ("a {P} boards a {B} a {F} rupees {A} will be levied", [
            ("P", ["waiting list passenger", "waiting list commuter", "waiting list traveller", "waiting list rail user", "waiting list person", "person with wl ticket"]),
            ("B", ["sleeper coach", "reserved ac coach", "ac coach", "ac compartment", "sleeper class", "reserved coach"]),
            ("F", ["fine of", "penalty of", "charge of", "financial penalty of", "monetary fine of"]),
            ("A", ["250", "300", "350", "440", "500", "750", "1000", "two hundred fifty", "three hundred", "four hundred forty", "five hundred"]),
        ]),
        ("baggage limit is {N} kg for {C} passengers", [
            ("N", ["30", "35", "40", "45", "50", "55", "60", "65", "70"]),
            ("C", ["sleeper class", "second class ac", "first class ac", "second class non ac", "ac coach", "third ac", "ac compartment"]),
        ]),
        ("in {CL} {R} lower berths per coach are reserved for {G}", [
            ("CL", ["sleeper class", "second ac", "third ac", "ac coach", "ac compartment", "sleeper coach", "first class ac"]),
            ("R", ["2", "3", "4", "5", "6", "7", "2 to 3", "3 to 4", "4 to 5", "5 to 6", "6 to 7", "two", "three", "four", "five", "six", "seven"]),
            ("G", ["senior citizens", "vulnerable groups", "women above forty five", "pregnant passengers", "persons with disabilities", "senior citizens and vulnerable groups", "women above forty five and pregnant passengers", "senior citizens and women", "women and pregnant women", "elderly passengers", "women", "seniors"]),
        ]),
        ("passengers can book railway tickets {D} days in advance", [
            ("D", ["30", "45", "60", "90", "120"]),
        ]),
        ("{P} get {Pct} concession in {C}", [
            ("P", ["senior citizens above sixty years", "seniors above sixty", "persons with disabilities", "students", "cancer patients", "dialysis patients", "blind passengers", "thalassemia patients", "war widows", "gallantry award winners", "differently abled", "widows", "ex servicemen"]),
            ("Pct", ["30", "40", "50", "60", "70", "thirty percent", "forty percent", "fifty percent", "sixty percent", "seventy percent"]),
            ("C", ["sleeper class", "second class", "second ac", "first class ac", "all classes", "third ac", "ac coach", "non ac", "third class"]),
        ]),
        ("if {T} is cancelled by indian railways passengers are entitled to {R}", [
            ("T", ["train", "express train", "mail train", "passenger train", "superfast train", "rajdhani express", "shatabdi express", "duronto express", "vande bharat"]),
            ("R", ["full refund", "full money back", "full fare refund", "full reimbursement", "complete refund", "entire fare back", "complete money back"]),
        ]),
        ("{C} can be registered on rail madad portal or by calling {H}", [
            ("C", ["complaints", "issues", "problems", "grievances", "feedback"]),
            ("H", ["139", "1-3-9", "one three nine", "1800-111-139", "9717630982"]),
        ]),
        ("a {P} who {A} shall be punished with {F}", [
            ("P", ["person", "passenger", "traveller", "commuter", "rail user", "individual"]),
            ("A", ["obstructs a railway servant in duty", "smokes in a compartment without consent", "enters a ladies compartment", "unlawfully enters upon a railway", "is in state of intoxication in coach", "suffers from infectious disease travels on railway", "refuses to leave reserved compartment", "creates nuisance in coach"]),
            ("F", ["fine", "penalty", "financial penalty", "monetary fine", "charge"]),
        ]),
        ("waiting list passengers {A}", [
            ("A", ["can travel in unreserved coaches only", "cannot board reserved coaches", "cannot enter major stations", "must wait at waiting areas", "can only travel general class", "cannot board ac or sleeper coaches", "must remain in unreserved coaches", "cannot travel reserved classes"]),
        ]),
        ("rac passengers {A}", [
            ("A", ["are entitled to a berth", "get a berth to sleep", "are permitted berth allocation", "can sleep during journey", "get priority berth", "get berth when available", "can occupy vacant berths"]),
        ]),
        ("passengers can call helpline {H} for {S}", [
            ("H", ["139", "1-3-9", "one three nine", "1800-111-139", "9717630982"]),
            ("S", ["complaints", "information", "enquiries", "assistance", "support", "help", "grievance redressal", "railway information", "train status"]),
        ]),
        ("the tte {A}", [
            ("A", ["cannot disturb sleeping passengers between 10pm and 6am", "can check tickets anytime during journey", "can allot vacant berths", "must provide bed rolls to ac passengers", "cannot refuse berth to rac passengers", "cannot levy unauthorized charges"]),
        ]),
        ("railways allots lower berth to {P}", [
            ("P", ["senior citizens", "women above forty five", "pregnant women", "persons with disabilities", "senior citizens and pregnant women", "women and senior citizens", "elderly passengers", "disabled passengers", "women and pregnant", "senior women"]),
        ]),
        ("tdr must be filed within {T}", [
            ("T", ["72 hours of arrival", "seventy two hours of arrival", "two days of arrival", "three days of arrival", "one week of arrival", "three days of train arrival"]),
        ]),
        ("pnr status can be checked on {W}", [
            ("W", ["irctc website", "rail mitra platform", "ntes app", "irctc app", "official railway website", "mobile app", "irctc mobile"]),
        ]),
        ("a {P} who travels beyond destination shall pay {C}", [
            ("P", ["passenger", "traveller", "commuter", "person"]),
            ("C", ["excess charge", "excess fare", "penalty charge", "difference fare plus penalty", "additional fare"]),
        ]),
        ("a {P} travelling in higher class than booked shall pay {C}", [
            ("P", ["passenger", "traveller", "commuter", "person"]),
            ("C", ["difference in fare", "fare difference", "excess fare", "penalty plus fare difference", "class difference amount"]),
        ]),
        ("passengers must carry {I} with ticket", [
            ("I", ["valid identity proof", "valid id card", "government id", "aadhaar card", "voter id", "passport", "driving license", "any government issued id"]),
        ]),
        ("if ac fails passengers get {R}", [
            ("R", ["refund of difference between ac and non ac", "fare refund", "reimbursement", "money back", "refund of fare difference"]),
        ]),
        ("duplicate ticket costs {A} rupees for {C}", [
            ("A", ["50", "100", "fifty", "one hundred"]),
            ("C", ["sleeper class", "sleeper", "second ac", "third ac", "ac coach", "first class"]),
        ]),
        ("{I} are prohibited on railways", [
            ("I", ["firearms", "flammable objects", "weapons", "hazardous substances", "explosives", "guns", "knives", "toxic chemicals", "inflammable materials"]),
        ]),
        ("children {A} travel {S}", [
            ("A", ["below five years", "under five", "less than five years", "upto five years"]),
            ("S", ["free", "without ticket", "free without ticket", "at no cost", "without charge"]),
        ]),
        ("children {A} pay {F}", [
            ("A", ["between five and twelve years", "aged five to twelve", "between 5 and 12 years", "5 to 12 years old"]),
            ("F", ["half fare", "half the price", "50 percent fare", "half ticket price"]),
        ]),
        ("students get {Pct} concession on {C}", [
            ("Pct", ["30", "40", "50", "thirty percent", "forty percent", "fifty percent"]),
            ("C", ["second class", "sleeper class", "all classes", "second class ac", "third class"]),
        ]),
        ("overcharging by pantry staff is {O}", [
            ("O", ["offence", "a crime", "punishable", "illegal", "against railway rules", "a violation"]),
        ]),
        ("{T} {F}", [
            ("T", ["rajdhani express", "shatabdi express", "duronto express", "garib rath", "vande bharat", "tejas express", "humsafar express", "maharajas express"]),
            ("F", ["have meals included in fare", "are ac three tier", "have aircraft style seats", "have executive chairs", "provide wifi service", "have onboard entertainment", "offer premium services"]),
        ]),
        ("{P} can get {A} at stations", [
            ("P", ["passengers", "elderly", "disabled", "differently abled", "senior citizens", "women"]),
            ("A", ["wheelchair assistance", "escort service", "help with luggage", "priority boarding", "special assistance"]),
        ]),
        ("if train is late by {T} passengers get {C}", [
            ("T", ["3 hours", "three hours", "4 hours", "more than 3 hours"]),
            ("C", ["full refund", "compensation", "free refreshments", "meal vouchers"]),
        ]),
        ("{P} not allowed in ladies coaches", [
            ("P", ["male passengers above 12 years", "men above twelve", "adult males", "boys above twelve years"]),
        ]),
        ("blind passengers {A}", [
            ("A", ["allowed guide dog in coach", "get fifty percent concession", "can avail escort service", "get priority seating"]),
        ]),
        ("passengers can book retiring room {A}", [
            ("A", ["at major stations in advance", "online", "at stations", "by paying nominal deposit"]),
        ]),
        ("{C} passengers get {T}", [
            ("C", ["ac class", "ac", "first class ac", "second ac", "third ac"]),
            ("T", ["complimentary bed rolls", "free bed rolls", "bed sheets provided", "blankets provided", "bedding provided"]),
        ]),
        ("{C} have charging points", [
            ("C", ["ac coaches", "ac compartments", "first class ac", "second ac", "premium trains"]),
        ]),
        ("{P} must show {I} with ticket", [
            ("P", ["passengers", "travellers", "commuters"]),
            ("I", ["valid id", "identity proof", "government id", "adhar card", "voter id"]),
        ]),
        ("cancellation charges are {A} percent of fare", [
            ("A", ["10", "20", "25", "30", "50"]),
        ]),
        ("passengers can upgrade to {C} by paying {A}", [
            ("C", ["higher class", "ac class", "ac coach", "first class"]),
            ("A", ["difference fare", "upgrade charge", "nominal upgrade fee"]),
        ]),
        ("group booking of {N} persons get {D} percent discount", [
            ("N", ["10", "15", "20", "25", "ten", "fifteen", "twenty", "twenty five"]),
            ("D", ["5", "10", "15", "ten percent", "fifteen percent"]),
        ]),
        ("{P} insurance available {A}", [
            ("P", ["passenger", "luggage", "accident"]),
            ("A", ["free with ticket", "at nominal cost", "online during booking", "optional at booking"]),
        ]),
        ("luggage carrier service available {A}", [
            ("A", ["at parcel offices", "at major stations", "for excess baggage", "at nominal charge"]),
        ]),
        ("bicycle booking allowed in brake van with {C}", [
            ("C", ["protection charges", "nominal fee", "insurance", "extra charge"]),
        ]),
        ("freight charges for {T} are {C}", [
            ("T", ["agricultural produce", "personal goods", "parcels", "household items", "commercial goods"]),
            ("C", ["as per weight", "nominal", "based on distance", "calculated per km"]),
        ]),
        ("{T} runs between {R}", [
            ("T", ["rajdhani express", "shatabdi express", "duronto express", "garib rath"]),
            ("R", ["delhi and mumbai", "mumbai and chennai", "kolkata and delhi", "bangalore and delhi", "delhi and kolkata", "mumbai and delhi"]),
        ]),
        ("platform ticket costs {A} rupees", [
            ("A", ["10", "20", "fifty", "one hundred", "25"]),
        ]),
        ("waiting room charges {A} for {C}", [
            ("A", ["included in ticket", "extra", "applicable", "nominal"]),
            ("C", ["reserved class passengers", "ac class", "sleeper class", "all passengers"]),
        ]),
        ("luggage responsibility lies with {P}", [
            ("P", ["passenger", "owner", "traveler", "passenger themselves"]),
        ]),
        ("emergency medical kit available with {P}", [
            ("P", ["train conductor", "tte", "train staff", "coach attendant"]),
        ]),
        ("first aid box provided in {C}", [
            ("C", ["every reserved coach", "all reserved coaches", "ac coaches", "sleeper class"]),
        ]),
        ("potable water available at {L}", [
            ("L", ["station counters", "platform water taps", "water booths", "coach water points"]),
        ]),
        ("doctor on call facility in {T}", [
            ("T", ["long distance trains", "premium trains", "rajdhani express", "all express trains"]),
        ]),
        ("ticket can be checked at {L}", [
            ("L", ["any point during journey", "platform entry", "coach entry", "during journey", "anytime"]),
        ]),
        ("ticket valid for {D}", [
            ("D", ["24 hours", "12 hours", "48 hours", "single journey", "one way only", "journey duration only"]),
        ]),
        ("booking confirmation via {C}", [
            ("C", ["sms", "email", "mobile app", "irctc website", "whatsapp"]),
        ]),
        ("payment accepted via {M}", [
            ("M", ["credit card", "debit card", "upi", "net banking", "wallet", "all methods"]),
        ]),
        ("charting done {T} before departure", [
            ("T", ["4 hours", "two hours", "30 minutes", "one hour"]),
        ]),
        ("{T} train has speed of {S} kmph", [
            ("T", ["rajdhani express", "shatabdi express", "vande bharat", "duronto express"]),
            ("S", ["130", "160", "180", "120", "100"]),
        ]),
        ("sleeper class has {N} berths per compartment", [
            ("N", ["6", "eight", "6 berths", "8 berths"]),
        ]),
        ("ac three tier has {N} berths per compartment", [
            ("N", ["8", "eight", "6", "9"]),
        ]),
        ("ac two tier has {N} berths per compartment", [
            ("N", ["4", "four", "6", "five"]),
        ]),
        ("ac first class has {N} berths per compartment", [
            ("N", ["2", "two", "4", "four"]),
        ]),
        ("tatkal booking opens at {T}", [
            ("T", ["10am for ac classes", "11am for sleeper", "day before departure", "one day prior"]),
        ]),
        ("premium tatkal costs {A} rupees extra", [
            ("A", ["15", "25", "30", "50", "twenty five rupees", "fifty rupees"]),
        ]),
    ]
    
    static_rules = [
        "railways provide lower berth quota regardless of concession usage",
        "passengers can extend journey by informing tte before destination",
        "no entry without valid ticket or pass",
        "a passenger without ticket pays excess charge plus penalty",
        "waiting list cannot board reserved coaches",
        "senior citizens get priority in lower berth allocation",
        "persons with disabilities get reserved berths in sleeper",
        "bag dimensions should not exceed 160 centimeters",
        "excess baggage charged at 1.5 times normal rate",
        "tatkal tickets non refundable if delay under 3 hours",
        "complaints handled through rail madad portal",
        "accident insurance provided free with ticket",
        "passenger insurance available online during booking",
        "blind passengers allowed guide dog in coach",
        "wheelchair assistance available at major stations",
        "retiring rooms bookable at stations in advance",
        "bed rolls provided free in ac classes",
        "non ac passengers can buy bed rolls from attendant",
        "charging points available in ac coaches",
        "wifi in select premium trains only",
        "every person shall be supplied with ticket on payment",
        "without ticket travel is punishable offence",
        "ticket checker can demand ticket anytime",
        "reserved coaches have allocated berths",
        "rac passengers share seats in reserved coaches",
        "lower berths are on side lower and side upper",
        "side lower berths are preferred by senior citizens",
        "meal options available in pantry car",
        "hot water available from pantry car",
        "tea and coffee sold in pantry car",
        "railway police can arrest ticketless travelers",
        "penalty for ticketless travel is six times fare",
        "train arrival time shown on station display",
        "coach position announced before arrival",
        "passengers should board before door closing",
        "luggage should be placed above seats",
        "heavy luggage placed in brake van",
        "seat numbers mentioned on ticket",
        "berth numbers allocated at booking",
        "window seat preferred by tourists",
        "aisle seat preferred by frequent travelers",
        "train runs on Indian Railway network",
        "broad gauge tracks in most sections",
        "electric traction on major routes",
        "diesel locomotives in remote areas",
        "train speed monitored by control room",
        "station master oversees train movements",
        "railway crossing gates operated manually",
        "unmanned crossings are dangerous",
        "passengers should stay inside coach",
        "dont lean out of moving train",
        "emergency window in every coach",
        "hammer to break glass in emergencies",
        "door should not be opened during motion",
        "toilet waste discharged on track",
        "bio toilet fitted in new trains",
        "ac temperature set by coach attendant",
        "fan speed adjustable in sleeper class",
        "lights dimmed during night hours",
        "announcements made in train",
        "station names announced before arrival",
        "next station shown on display",
        "wifi password in premium trains",
        "meal served at scheduled time",
        "veg non veg meals available",
        "meal preference noted at booking",
        "water bottle allowed inside coach",
        "food from outside allowed",
        "eating in coach is permitted",
        "spitting in coach is prohibited",
        "cleanliness to be maintained",
        "dustbin provided in every coach",
        "waste to be thrown in dustbin",
        "mobile phones allowed in coach",
        "loud music not allowed in coach",
        "headphones to be used for audio",
        "smoking strictly prohibited",
        "tobacco chewing not allowed",
        "alcohol possession is banned",
        "gambling in coach is offence",
        "begging in train is prohibited",
        "distributing pamphlets not allowed",
        "politics campaigning banned in trains",
        "religious preaching regulated",
        "photography restricted in stations",
        "selfie on tracks is dangerous",
        "trespassing on tracks is illegal",
        "animals on tracks cause accidents",
        "railway property should not be damaged",
        "graffiti on walls is crime",
        "signage to be followed",
        "platform walking restricted",
        "climbing moving train is fatal",
        "jumping from train causes injury",
        "railway helpline available 24x7",
        "railway police for security",
        "women helpline for emergencies",
        "child security in stations",
        "lost and found at stations",
        "complaints feedback mechanism exists",
        "refund processed within 7 days",
        "e ticket easier than counter ticket",
        "counter ticket requires id proof",
        "tatkal booking has limited quota",
        "premium tatkal costs extra",
        "general quota open for all",
        "defence quota for army personnel",
        "cancer patients get certificate from hospital",
        "medical certificate needed for concession",
        "concession form available at stations",
        "concession need valid id proof",
        "war widow needs certificate",
        "freedom fighter needs certificate",
        "parliamentarian quota exists",
        "mp mla quota for travel",
        "senior citizen concession automatic",
        "age proof needed for senior concession",
        "student concession requires institution letter",
        "rural health certificate acceptable",
        "national awardees need award copy",
        "police verification for luggage",
        "customs declaration for valuables",
        "jewellery allowed within limits",
        "gold limit as per customs rule",
        "electronics should be declared",
        "laptop for personal use allowed",
        "professional equipment for work allowed",
        "sports equipment as luggage",
        "musical instrument as hand luggage",
        "pet animal allowed in cage",
        "pet booking requires health certificate",
        "livestock not allowed in passenger coach",
        "birds in cage allowed",
        "snake or reptile not allowed",
        "ferrying cattle is offence",
        "illegal wildlife trade banned",
        "hand baggage should not exceed 7 kg",
        "personal belongings at own risk",
        "luggage to be locked by passenger",
        "unclaimed luggage destroyed after notice",
        "luggage insurance recommended",
        "fragile items not accepted for booking",
        "liquid items need special packing",
        "computers sensitive to handling",
        "glass items need special packaging",
        "perishable goods not accepted",
        "live plants allowed with certificate",
        "dry fruits can be carried as luggage",
        "food grains allowed in limited quantity",
        "compressed gas cylinders banned",
        "matches allowed in small quantity",
        "cigarette lighters limited quantity",
        "perfumes limited to 100 ml",
        "aerosol cans not allowed",
        "paint containers not allowed",
        "acid containers strictly banned",
        "batteries to be carried as hand luggage",
        "lithium batteries need declaration",
        "power banks limited to 100 wh",
        "mobile charger allowed",
        "laptop charger allowed",
        "extension cord allowed",
        "electric kettle not allowed",
        "iron not allowed in luggage",
        "heater not allowed in coach",
        "burner not allowed in coach",
        "cooking in coach prohibited",
        "dhobi allowed on select trains",
        "laundry service on select trains",
        " newspapers provided in ac coaches",
        "magazines available in pantry car",
        "railway complaint book in every coach",
        "suggestion box at stations",
        "feedback form available online",
        "customer care number displayed",
        "station manager available for complaints",
        "compensation for delayed trains",
        "complaint redressal within 30 days",
        "refund for cancelled tickets automatic",
        "failed transaction reversal within 7 days",
        "multiple booking single cancellation",
        "bulk booking requires agent id",
        "travel agent commission available",
        "corporate booking facility exists",
        "special quota for pilgrimage sites",
        "khela baba express for devotees",
        "sabri express for saint followers",
        "ayana train for nepal pilgrimage",
        "amritsar express for religious travel",
        "ayana janmabhoomi train for locals",
        "shikharji train for mountain visit",
        "amarnath yatra train for kashmir",
        "char dhams train for holy sites",
        "museum train for heritage tourism",
        "toy train for hill stations",
        "heritage railway runs on weekends",
        "narrow gauge for mountain routes",
        "meter gauge in remote areas",
        "train number has four digits",
        "station code has three letters",
        "pnr number has ten digits",
        "ticket number unique identifier",
        "coach position from engine",
        "a coach is also called bogie",
        "locomotive also called engine",
        "guard van at end of train",
        " brake van for luggage and guards",
        "power car supplies electricity",
        "end coach is guard compartment",
        "pantograph collects electricity",
        "overhead wire carries power",
        "signal aspects control movement",
        "green signal means proceed",
        "yellow signal means caution",
        "red signal means stop",
        "semaphore signals in old sections",
        "electronic interlocking in new sections",
        "platform length for 24 coach trains",
        "platform height matches floor level",
        "foot overbridge connects platforms",
        "escalators at major stations",
        "lifts for elderly and disabled",
        "waiting hall on each platform",
        "cloak room facility at stations",
        "short stay rooms available",
        "package tour services available",
        "car rental booking at stations",
        "auto booking at station exit",
        "taxi booking at station exit",
        "food plaza at major stations",
        "cafe outlets at stations",
        "book stall at platform",
        "newspaper stall at stations",
        "fruit stall at stations",
        "water bottle vendor at stations",
        "tea stall at platform",
        "railway mela held annually",
        "railway museum open to public",
        "railway park for recreation",
        "night coach timing 9pm to 6am",
        "ladies coach timing flexible",
        "handicapped compartment near gate",
        "dining car in select trains",
        "shared table in pantry car",
        "meal timing 8am 12pm 8pm",
        "breakfast served morning hours",
        "lunch served afternoon",
        "dinner served evening",
        "vegetarian meals available",
        "jain meals on request",
        "halal food available",
        "customized diet on medical ground",
        "tea coffee available all hours",
        "mineral water sold in coach",
        "cold drinks available",
        "ice cream sold in pantry car",
        "snacks available in pantry",
        "railway cake popular item",
        "samosa served in pantry",
        "dosa made in pantry car",
        "railway biryani famous",
        "meals affordable price",
        "voucher system in pantry",
        "cashless payment in pantry",
        "upi accepted in pantry car",
        "card payment accepted in pantry",
    ]
    
    for r in static_rules:
        seen.add(r)
        rules.append(r)
    
    for template in templates:
        tmpl, options = template
        values = [v for k, v in options]
        
        for combo in product(*values):
            d = dict(zip([k for k, v in options], combo))
            try:
                rule = tmpl.format(**d)
                rule = ' '.join(rule.split())
                if rule and rule not in seen and len(rule) > 15:
                    rules.append(rule)
                    seen.add(rule)
            except:
                pass
    
    random.shuffle(rules)
    return rules[:10000]

if __name__ == "__main__":
    rules = create_augmented_rules()
    print(f"Generated {len(rules)} unique rules")
    
    with open("railway_data.txt", "w") as f:
        for rule in rules:
            f.write(f"{rule}\n")
    
    print(f"Saved {len(rules)} lines")
    print("\nSample:")
    for r in rules[:20]:
        print(f"  {r}")