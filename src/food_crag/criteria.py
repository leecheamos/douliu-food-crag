"""Conservative local parsing: incomplete/contradictory turns never change state."""
import re
import unicodedata
from dataclasses import dataclass, replace

from .domain import SearchCriteria

CUISINES = {
    "麵食": ("麵食", "麵類", "麵店", "麵"),
    "牛肉麵": ("牛肉麵",), "鍋燒意麵": ("鍋燒意麵",),
    "拉麵": ("拉麵",), "義大利麵": ("義大利麵",),
    "火鍋": ("火鍋", "鍋物"), "牛排": ("牛排",),
    "日式料理": ("日式料理", "日本料理", "日式"),
    "義式料理": ("義式料理", "義式"),
    "台菜": ("台菜", "臺菜"), "熱炒": ("熱炒",),
    "早午餐": ("早午餐",), "飯食": ("飯食", "飯類", "飯"),
    "肉燥飯": ("肉燥飯", "滷肉飯", "魯肉飯"),
    "豬腳飯": ("豬腳飯",), "菜飯": ("菜飯",), "蛋飯": ("蛋飯",),
    "生魚片": ("生魚片",), "碗粿": ("碗粿",), "羹麵": ("羹麵",),
    "魚羹": ("魚羹",), "魷魚羹": ("魷魚羹",),
    "羊肉料理": ("羊肉料理", "羊肉"), "海鮮": ("海鮮",),
    "刈包": ("刈包",), "簡餐": ("簡餐",),
    "肉圓": ("肉圓",), "和風洋食": ("和風洋食",),
}
ALIASES = {alias: canonical for canonical, aliases in CUISINES.items() for alias in aliases}
PARENTS = {
    "牛肉麵": "麵食", "鍋燒意麵": "麵食", "拉麵": "麵食", "義大利麵": "麵食", "羹麵": "麵食",
    "肉燥飯": "飯食", "豬腳飯": "飯食", "菜飯": "飯食", "蛋飯": "飯食",
}
LANDMARKS = {
    "雲林科技大學": ("雲林科技大學", "雲科大"),
    "斗六車站": ("斗六火車站", "斗六車站"),
    "雲林縣政府": ("雲林縣政府",),
    "鎮南國小站": ("鎮南國小站",),
}
LANDMARK_ALIASES = {alias: name for name, aliases in LANDMARKS.items() for alias in aliases}


def alternatives(values):
    return "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))


CUISINE_PATTERN = re.compile(
    r"(?P<prefix>取消排除|解除排除|不要吃|不想吃|不吃|不要|排除|也想吃|加入|想吃|改吃|換成|吃)?"
    r"\s*(?P<name>" + alternatives(ALIASES) + r")(?P<suffix>也可以|也行)?"
)
BUDGET_PATTERN = re.compile(
    r"(?:預算\s*(?:改成|改為|改到|是|為|上限|不超過|最多)?\s*|(?:不超過|最多|上限)\s*)(?P<a>\d+)\s*(?:元|塊)?(?:以內|以下)?"
    r"|(?P<b>\d+)\s*(?:元|塊)(?:以內|以下)?"
    r"|(?P<c>\d+)\s*(?:以內|以下)"
)


@dataclass(frozen=True)
class ParseResult:
    criteria: SearchCriteria
    questions: tuple[str, ...] = ()
    reset: bool = False


class RuleCriteriaParser:
    def parse(self, message: str, previous: SearchCriteria | None = None) -> ParseResult:
        original = previous or SearchCriteria()
        text = unicodedata.normalize("NFKC", message).strip()
        if re.fullmatch(r"(?:重設|重新開始|清除條件|reset)[\s。!！]*", text, re.IGNORECASE):
            return ParseResult(SearchCriteria(), reset=True)
        current = original
        changed = False
        questions = []

        def consume(pattern, callback):
            nonlocal text, changed
            regex = re.compile(pattern) if isinstance(pattern, str) else pattern

            def apply(match):
                nonlocal changed
                changed = True
                callback(match)
                return " "

            text = regex.sub(apply, text)

        def clear(field, value):
            def apply(match):
                nonlocal current
                current = replace(current, **{field: value})
            return apply

        consume(r"取消預算(?:限制)?|預算不限|不限預算", clear("budget_twd", None))
        consume(r"取消料理(?:偏好|限制)?|料理不限|不限料理", clear("cuisines", ()))
        consume(r"取消排除(?:條件|偏好|限制)(?=\s*(?:$|[，,。；;]))|取消排除(?=\s*(?:$|[，,。；;]))", clear("excluded_cuisines", ()))
        consume(r"取消(?:地標|地點)(?:限制)?|不限地點", clear("landmark", None))
        consume(r"取消(?:時段|用餐時段)(?:限制)?|時段不限", clear("meal", None))

        budgets = []
        consume(BUDGET_PATTERN, lambda m: budgets.append(next(v for v in m.groups() if v is not None)))
        if budgets:
            if len(budgets) != 1 or len(budgets[0]) > 9 or int(budgets[0]) <= 0:
                questions.append("請提供一個大於 0 的整數預算上限，例如「預算 200 元」。")
            else:
                current = replace(current, budget_twd=int(budgets[0]))

        meals = []
        # 早午餐 is a cuisine label, not a confirmed lunch service period.
        consume(r"(?:改成|改為|換成)?(?<!早)(?:午餐|中餐|中午|午飯)", lambda m: meals.append("lunch"))
        consume(r"(?:改成|改為|換成)?(?:晚餐|晚上|晚飯)", lambda m: meals.append("dinner"))
        if len(set(meals)) > 1:
            questions.append("這次要找午餐還是晚餐？請選一個時段。")
        elif meals:
            current = replace(current, meal=meals[0])

        places = []
        consume(alternatives(LANDMARK_ALIASES), lambda m: places.append(LANDMARK_ALIASES[m[0]]))
        if len(set(places)) > 1:
            questions.append("這次要以哪一個地標為主？請保留一個地標。")
        elif places:
            current = replace(current, landmark=places[0])
        consume(r"雲林縣|斗六市|斗六", lambda m: None)
        if "附近" in text and current.landmark is None:
            questions.append("請指定地標，例如「斗六車站附近」；目前不使用定位或計算距離。")

        positive, negative, remove_excluded = [], [], []
        additive = False
        last_end, last_negative = 0, False
        cuisine_text = text

        def cuisine(match):
            nonlocal last_end, last_negative, additive
            prefix, suffix = match["prefix"], match["suffix"]
            name = ALIASES[match["name"]]
            between = cuisine_text[last_end:match.start()].strip()
            inherited_negative = last_negative and bool(re.fullmatch(r"(?:和|及|、|或|跟|還有)+", between))
            if last_negative and not prefix and not suffix and not inherited_negative:
                questions.append("排除條件後的料理意思不明，請明說「不要」或「想吃」。")
            if suffix and prefix in {"不要吃", "不想吃", "不吃", "不要", "排除"}:
                questions.append("同一句包含排除與「也可以」，請確認要排除還是取消排除。")
            if prefix in {"取消排除", "解除排除"} or suffix:
                remove_excluded.append(name)
                last_negative = False
            elif prefix in {"不要吃", "不想吃", "不吃", "不要", "排除"} or (not prefix and inherited_negative):
                negative.append(name)
                last_negative = True
            else:
                if not positive:
                    additive = prefix in {"加入", "也想吃"}
                positive.append(name)
                last_negative = False
            last_end = match.end()

        consume(CUISINE_PATTERN, cuisine)
        exclusions = [x for x in current.excluded_cuisines if x not in remove_excluded]
        exclusions = list(dict.fromkeys(exclusions + negative))
        cuisines = list(current.cuisines)
        if positive:
            cuisines = list(dict.fromkeys((cuisines if additive else []) + positive))
        current = replace(current, cuisines=tuple(cuisines), excluded_cuisines=tuple(exclusions))
        conflicts = [x for x in cuisines if x in exclusions or PARENTS.get(x) in exclusions]
        if conflicts:
            questions.append("料理偏好與排除條件衝突：" + "、".join(conflicts) + "。請先取消排除或修改料理偏好。")
        if set(negative) & set(remove_excluded):
            questions.append("同一句同時新增及取消相同排除條件，請確認要保留哪個設定。")
        if any(PARENTS.get(name) in exclusions for name in remove_excluded):
            questions.append("取消單一料理的排除仍受整類排除限制，請先修改整類排除條件。")

        # Only grammatical filler is ignored. Unknown requests trigger a full
        # rollback, so e.g. allergies, distances and negations are never dropped.
        remainder = re.sub(
            r"(?:請問|幫我|請|我|要|找|推薦|餐廳|吃什麼|有什麼|在|的|附近|這次|今天|可以嗎|謝謝|和|及|或|跟|還有|也|都|以|為主)|[\s,，。.!！?？、:：;；]",
            "", text,
        )
        if remainder:
            questions.append(f"尚無法理解「{remainder}」。目前支援預算、料理／排除偏好、午晚餐、斗六及指定地標，請換個說法。")
        elif not changed:
            questions.append("請告訴我料理或條件，例如「200 元以內，不要火鍋」。")
        if questions:
            return ParseResult(original, tuple(questions))
        return ParseResult(current)
