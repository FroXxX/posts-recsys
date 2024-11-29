import re
import dateparser


class DatePreprocessor:

    ordinal_to_number = {
        "first": "1", "second": "2", "third": "3",
        "fourth": "4", "fifth": "5", "sixth": "6",
        "seventh": "7", "eighth": "8", "ninth": "9",
        "tenth": "10", "eleventh": "11", "twelfth": "12",
        "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
        "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
        "nineteenth": "19", "twentieth": "20", "twenty-first": "21",
        "twenty-second": "22", "twenty-third": "23", "twenty-fourth": "24",
        "twenty-fifth": "25", "twenty-sixth": "26", "twenty-seventh": "27",
        "twenty-eighth": "28", "twenty-ninth": "29", "thirtieth": "30",
        "thirty-first": "31"
    }

    ordinals = [
        "first", "second", "third", "fourth",
        "fifth", "sixth", "seventh", "eighth",
        "ninth", "tenth", "eleventh", "twelfth",
        "thirteenth", "fourteenth", "fifteenth",
        "sixteenth", "seventeenth", "eighteenth",
        "nineteenth", "twentieth", "twenty-first",
        "twenty-second", "twenty-third", "twenty-fourth",
        "twenty-fifth", "twenty-sixth", "twenty-seventh",
        "twenty-eighth", "twenty-ninth", "thirtieth", "thirty-first"
    ]

    months = [
        "january", "february", "march", "april",
        "may", "june",  "july", "august", "september",
        "october", "november", "december"
    ]

    months_brief = [
        "jan", "feb", "mar", "apr", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    def __init__(self, n2w_engine):
        self.n2w = n2w_engine

    def _replace_matched_date_w_day(self, match):
        found_date = match.group(0)
        date = dateparser.parse(found_date)
        if not date:
            chk = "(?:{})".format("|".join(self.ordinal_to_number.keys()))
            ordinal = re.search(chk, found_date)

            if not ordinal:
                return ""

            tmp = found_date.replace(
                ordinal.group(0), self.ordinal_to_number[ordinal.group(0)]
            )
            date = dateparser.parse(tmp)
        if not date:
            return ""
        day = self.n2w.ordinal(date.day)
        new_str_for_date = (
            f" {self.n2w.number_to_words(day)}"
            f" of {date.strftime('%B').lower()}"
            f" {self.n2w.number_to_words(date.year)} "
        )
        return new_str_for_date

    def _replace_matched_date_wo_day(self, match):
        found_date = match.group(0)
        date = dateparser.parse(found_date)
        new_str_for_date = (
            f" {date.strftime('%B').lower()}"
            f" {self.n2w.number_to_words(date.year)} "
        )
        return new_str_for_date

    def find_and_normalise_dates(self, text):

        ordinal_pattern = r"\b(?:" + "|".join(self.ordinals) + r")\b"

        month_pattern = (
            r"\b(?:(?:"
            + "|".join(self.months)
            + r")\b|(?:"
            + "|".join(self.months_brief)
            + r")\b\.?)"
        )

        date_pattern_wo_day = (
            r"""
            # Month-Year
            (?:
                """
            + month_pattern
            + r"""(?:[, ]+\d{4})
            )
            |
            (?:
                \b\d{2}/\d{4}\b
            )
        """
        )

        date_pattern_w_day = (
            r"""
            # Day Month Year
            (?:
                \d{1,2}(?:st|nd|rd|th)?(?:[ ]+of)?[ ]+
                """
            + month_pattern
            + r"""
                (?:[,\s@]*\d{4})?\b
            )
            |
            # Year-Month-Day
            (?:
                \b\d{4}(?P<ymd>[-/.])\d{1,2}(?P=ymd)\d{1,2}\b
            )
            |
            # Day/Month/Year or Day/Month
            (?:
                \b\d{1,2}/\d{1,2}(?:/(\d{2}|\d{4}))?\b
            )
            |
            # Day-Month-Year or Day.Month.Year
            (?:
                \b\d{1,2}(?P<dmy>[.-])\d{1,2}(?:(?P=dmy)(\d{2}|\d{4}))\b
            )
            |
            # Month Day Year
            (?:
                """
            + month_pattern
            + r"""[ ]+
                \d{1,2}(?:st|nd|rd|th)?
                (?:[,\s]+\d{4})?\b
            )
            |
            # Ordinal-Day-Month-Year
            (?:
                """
            + ordinal_pattern
            + r"""
                (?:[ ]+of)?[ ]+
                """
            + month_pattern
            + r"""
                (?:[,\s@]*\d{4})?\b
            )
            |
            # Month Ordinal
            (?:
                """
            + month_pattern
            + r"""
                [ ]+
                """
            + ordinal_pattern
            + r"""
                (?:[,\s]+\d{4})?\b
            )

        """
        )

        modified_text = re.sub(
            date_pattern_w_day, self._replace_matched_date_w_day, text, flags=re.VERBOSE
        )

        modified_text = re.sub(
            date_pattern_wo_day,
            self._replace_matched_date_wo_day,
            modified_text,
            flags=re.VERBOSE,
        )

        return modified_text
