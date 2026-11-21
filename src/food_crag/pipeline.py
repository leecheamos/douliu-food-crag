"""Parse conditions, retrieve, evaluate, then reply with grounded recommendations."""
from collections import Counter
from threading import RLock

from .conversation import InMemoryConversationStore
from .criteria import CUISINES, RuleCriteriaParser
from .domain import ChatResult
from .dataset import load_restaurants
from .retriever import RestaurantRetriever, ranking_query
from .config import Settings
from .adapters.gemini import GeminiLanguageModel
from .embedding import load_from_settings
from .evaluator import GeminiEvaluator, retrieve_and_evaluate
from .llm_criteria import INSTRUCTION as PARSE_INSTRUCTION, GeminiCriteriaParser
from .recommendation import TemplateReplyGenerator, verify_recommendation

GUIDE_TEXT = "我還不太確定你想吃什麼～想吃飯、麵，還是其他？點下面的按鈕，或直接跟我說預算也可以！"
BUDGET_BUTTONS = ("100元以內", "200元以內", "不限預算")


class RecommendationPipeline:
    def __init__(self, retriever=None, settings=None, model=None, reply_generator=None, parser_model=None):
        self.parser = RuleCriteriaParser()
        self.store = InMemoryConversationStore()
        self._turn_lock = RLock()
        settings = settings or Settings.from_env()
        if retriever is None:
            restaurants = load_restaurants()
            # Missing/stale cache disables embeddings instead of failing the chat.
            semantic, self.embedding_status = load_from_settings(restaurants, settings)
            retriever = RestaurantRetriever(restaurants, semantic)
        else:
            self.embedding_status = "hybrid_rrf" if retriever.semantic is not None else "disabled"
        self.retriever = retriever
        self.evaluator = GeminiEvaluator(self.retriever, model if model is not None else GeminiLanguageModel(settings))
        self.catalog = {r.id: r for r in self.retriever.restaurants}
        self.fallback = TemplateReplyGenerator(self.retriever.restaurants)
        self.reply_generator = reply_generator if reply_generator is not None else self.fallback
        # Free-form fallback. An injected evaluation model (tests) means rules only
        # unless a parser model is injected too.
        if parser_model is None and model is None:
            parser_model = GeminiLanguageModel(settings, PARSE_INSTRUCTION)
        self.free_parser = GeminiCriteriaParser(parser_model) if parser_model is not None else None
        # Guide buttons: the most common cuisines in the data, each a phrase the rule parser reads.
        counts = Counter(c for r in self.retriever.restaurants for c in r.cuisines if c in CUISINES)
        # 飯食/麵食 first because the guide sentence asks "rice, noodles or something else?".
        top = [c for c in ("飯食", "麵食") if counts[c]] + [c for c, _ in counts.most_common() if c not in ("飯食", "麵食")]
        self.guide_buttons = (*(f"想吃{c}" for c in top[:3]), "都可以")

    def handle(self, user_id: str, message: str) -> ChatResult:
        if not isinstance(user_id, str) or not user_id.strip() or not isinstance(message, str) or not message.strip():
            raise ValueError("user_id and message must be nonempty strings")
        # Serialize read/parse/write to prevent concurrent requests losing updates.
        with self._turn_lock:
            previous = self.store.get(user_id)
            result = self.parser.parse(message, previous)
            if result.unparsed and self.free_parser is not None:
                result = self.free_parser.parse(message, previous) or result
            if result.guide or result.unparsed:
                # Off-topic, undecided, or unreadable with the model unavailable: guide, don't refuse.
                guide = result.questions[0] if result.guide else GUIDE_TEXT
                if any(r.name in guide for r in self.retriever.restaurants):
                    guide = GUIDE_TEXT  # Guide text must never recommend a shop on its own.
                return ChatResult(status="guidance", reply=guide, criteria=previous or result.criteria,
                                  quick_replies=self.guide_buttons)
            if result.questions:
                status = "clarification_required"
            elif result.reset:
                self.store.reset(user_id)
                status = "reset"
            else:
                self.store.set(user_id, result.criteria)
                status = "criteria_updated"
        c = result.criteria
        meal = {"lunch": "午餐", "dinner": "晚餐", None: "不限"}[c.meal]
        summary = (
            f"目前條件：地區 {c.county}{c.district}；時段 {meal}；"
            f"預算 {'不限' if c.budget_twd is None else str(c.budget_twd) + ' 元以內'}；"
            f"料理 {'、'.join(c.cuisines) or '不限'}{'（' + '、'.join(c.dishes) + '）' if c.dishes else ''}；"
            f"排除 {'、'.join(c.excluded_cuisines) or '無'}；地標 {c.landmark or '不限'}。"
        )
        if result.questions:
            reply = "本輪未修改條件。" + " ".join(result.questions) + "\n" + summary
        else:
            reply = ("已清除你的條件。" if result.reset else "已記錄你的條件。") + summary
        retrieval = None
        evaluation = None
        attempts = ()
        restaurant_ids = ()
        if status == "criteria_updated":
            retrieval, evaluation, attempts = retrieve_and_evaluate(self.retriever, self.evaluator, c, ranking_query(c))
            try:
                recommendation = self.reply_generator.generate(c, retrieval, evaluation)
            except Exception:  # A failing generator must still leave a readable reply.
                recommendation = None
            if recommendation is None or not verify_recommendation(recommendation, self.catalog, evaluation, retrieval):
                recommendation = self.fallback.generate(c, retrieval, evaluation)
                if not verify_recommendation(recommendation, self.catalog, evaluation, retrieval):
                    recommendation = self.fallback.unavailable()
            detail = recommendation.reply
            restaurant_ids = recommendation.restaurant_ids
        else:
            detail = "本輪未執行檢索。"
        # A recommendation speaks for itself; the condition summary is only
        # prepended when no restaurants are listed.
        return ChatResult(
            status=status,
            reply=detail if restaurant_ids else reply + "\n" + detail,
            restaurant_ids=restaurant_ids,
            criteria=c,
            questions=result.questions,
            retrieval=retrieval,
            evaluation=evaluation,
            attempts=attempts,
            # Second guiding step: offer a budget once shops are listed without one.
            quick_replies=BUDGET_BUTTONS if restaurant_ids and c.budget_twd is None else (),
        )
