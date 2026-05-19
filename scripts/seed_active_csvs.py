"""One-off: migrate inline word constants from dissman.py into CSVs.

Two output files: insults/active/adjectives.csv and insults/active/nouns.csv,
each with columns word,category,added_date. Duplicates are preserved so that
draws weighted by frequency work naturally.
"""
import csv
from pathlib import Path

g_adj = "clumsy scatterbrained grumpy sloppy cranky loony cheeky stubborn sneaky rascally mopey shifty snarky pouty grungy fussy sassy zonked knobby topsy-turvy clumsy scatterbrained grumpy sloppy cranky loony cheeky stubborn sneaky rascally mopey shifty snarky pouty grungy fussy sassy zonked knobby topsy-turvy"
g_nouns = "doodle hamburger backpack bedding bedspread binder blanket blinds bookcase book broom brush bucket calendar angler toad horse candle carpet chair china clock coffee-table comb comforter computer container couch credenza curtain cushion heater houseplant magnet mop radiator radio refrigerator rug saucer saw scissors screwdriver settee shade sheet shelf shirt shoe smoke-detector sneaker socks sofa speaker toy tool tv toothpaste towel nutcase toaster pancake muffin wombat caboose goblin pirate ninja meatball cupcake tadpole dingbat noodle turnip alien gadget grasshopper pickle wigwam bonnethead sharksucker"
r_adj = "shitty great-value pick-me horsefaced cum-guzzling christian vaginal straight cum semen smegma discharge fallopian anal aggressive arrogant boastful bossy boring careless clingy cruel cowardly deceitful dishonest greedy harsh impatient impulsive jealous moody narrow-minded overcritical rude selfish untrustworthy unhappy cumguzzling unfuckable incestuous sick perverted deranged depraved mountain-dew-drinking butterfaced self-centered revolting repellent repulsive sickening nauseating nauseous stomach-churning stomach-turning off-putting unpalatable unappetizing uninviting unsavoury distasteful foul nasty obnoxious odious"
r_nouns = "eunuch no-dick cum-for-brains toesniffer dicksucker dicksniffer toesucker simp skidmark shit-stain anal-fissure anal-wart anal-cyst vaginal-cyst vaginal-discharge smegma foreskin dick-cheese cumguzzler yuppy hippy karen boomer dork nerd dweeb unfuckable pedophile butterface needledick incel neckbeard wart genital-wart homewrecker doof douche doucheholster douchebag cum-receptacle cum-dumpster cumrag cumslut bum degenerate derelict good-for-nothing no-account no-good slacker hetrosexual buttlover breitbart-reader andriod-user trump-lover republican cumslut buttmuncher nutsack ballsack boner christian penis cunt twat asshole fucker shitbag shit-for-brains cumrag gland intestine cecum colon rectum liver gallbladder mesentery pancreas anus kidney ureter bladder urethra ovary tube uterus cervix discharge vagina"
old_adj = "froward pernickety laggardly moonstruck mumpsimus spleeny fribble dandiprat rattlecap slugabed cacafuego raggabrash dithering muddle-headed tatterdemalion claptrap wifty bedswerver lackadaisical flapdoodle"
old_nouns = "scallywag naysayer neerdowell landlover fustilarian snollygoster popinjay lickspittle rakefire whippersnapper noodle mumblecrust zounderkite gillywetfoot doodle pettifogger fopdoodle mooncalf clodpole hugger-mugger ragamuffin scalawag ninnyhammer flapdragon"

ADJ = [
    ("g", g_adj.split()),
    ("r", r_adj.split()),
    ("old", old_adj.split()),
]
NOUN = [
    ("g", g_nouns.split()),
    ("r", r_nouns.split()),
    ("old", old_nouns.split()),
]

ADDED = "2026-05-18"


def _write_active(path: Path, groups: list[tuple[str, list[str]]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "category", "added_date"])
        for category, words in groups:
            for word in words:
                if word:
                    w.writerow([word, category, ADDED])
                    count += 1
    return count


def _write_submission_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(["word", "category", "submission_date"])


def main():
    root = Path(__file__).resolve().parent.parent

    n_adj = _write_active(root / "insults" / "active" / "adjectives.csv", ADJ)
    print(f"wrote insults/active/adjectives.csv ({n_adj} rows)")

    n_noun = _write_active(root / "insults" / "active" / "nouns.csv", NOUN)
    print(f"wrote insults/active/nouns.csv ({n_noun} rows)")

    _write_submission_header(root / "insults" / "submissions" / "adjectives.csv")
    _write_submission_header(root / "insults" / "submissions" / "nouns.csv")
    print("wrote submission headers")


if __name__ == "__main__":
    main()
