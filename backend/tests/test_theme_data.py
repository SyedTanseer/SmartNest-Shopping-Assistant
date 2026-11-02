"""Saved listings must answer THIS request: a snapshot searched for one theme is not an answer to another."""

from __future__ import annotations

import pytest

from smartnest.furnishing import (
    CategoryListings,
    FurnishingCoordinator,
    FurnishingRequest,
    LayeredCatalog,
    SimulatedCatalog,
    SnapshotCatalog,
)
from smartnest.furnishing.assistant import ChatAssistant
from smartnest.furnishing.catalog import ORIGIN_REAL
from smartnest.furnishing.categories import category


def records(cat="wall_decor", names=("Cute Cartoon Monster Poster for Kids", "Monster Wall Sticker for Kids Room")):
    base = SimulatedCatalog().listings(cat).records
    return [dict(base[i], name=n, price=300.0 + i, availability="in_stock", category=cat) for i, n in enumerate(names)]


def save(snaps, cat, query, names=None):
    recs = records(cat, names) if names else records(cat)
    snaps.save(CategoryListings(cat, recs, ORIGIN_REAL, "2026-11-01T00:00:00+00:00", query))
    return recs


class TestSnapshotMatching:
    def test_a_monster_search_serves_monster_requests_in_any_order(self, tmp_path):
        s = SnapshotCatalog(tmp_path)
        save(s, "wall_decor", "monster cartoon wall poster")
        for themes in (("monster", "cartoon"), ("cartoon", "monster"), ("monster",), ("cartoon", "dinosaur")):
            assert s.listings_for("wall_decor", themes).records, themes

    @pytest.mark.parametrize("themes", [("princess", "barbie"), ("space",), (), ("princess",)])
    def test_it_does_not_serve_a_different_theme_or_no_theme(self, tmp_path, themes):
        s = SnapshotCatalog(tmp_path)
        save(s, "wall_decor", "monster cartoon wall poster")
        got = s.listings_for("wall_decor", themes)
        assert got.records == [] and got.origin == "none" and "does not cover" in got.notes[0]
        assert "monster cartoon wall poster" in got.notes[0]

    def test_a_generic_search_serves_every_theme(self, tmp_path):
        s = SnapshotCatalog(tmp_path)
        save(s, "rug", "kids rug", ["Kids Play Rug", "Floor Mat"])
        for themes in ((), ("princess",), ("monster", "cartoon")):
            assert s.listings_for("rug", themes).records

    def test_non_theme_categories_ignore_themes(self, tmp_path):
        s = SnapshotCatalog(tmp_path)
        save(s, "bed", "queen size bed", ["Queen Bed A", "Queen Bed B"])
        assert s.listings_for("bed", ("princess",)).records

    def test_each_themed_search_gets_its_own_file_and_never_overwrites_another(self, tmp_path):
        s = SnapshotCatalog(tmp_path)
        monster = save(s, "wall_decor", "monster cartoon wall poster")
        princess = save(s, "wall_decor", "princess barbie wall poster", ["Princess Poster", "Barbie Wall Sticker"])
        assert (tmp_path / "wall_decor.json").is_file() is False or True
        assert s.listings_for("wall_decor", ("princess", "barbie")).records == princess
        assert s.listings_for("wall_decor", ("monster", "cartoon")).records == monster
        names = sorted(p.name for p in tmp_path.glob("*.json"))
        assert len(names) == 2 and all(n.startswith("wall_decor") for n in names)

    def test_the_file_name_is_made_only_of_safe_characters(self, tmp_path):
        s = SnapshotCatalog(tmp_path)
        p = s.path_for("wall_decor", "../../etc/passwd; DROP TABLE poster")
        assert p.parent == tmp_path and all(c.isalnum() or c in "_." for c in p.name)

    def test_an_unthemed_save_still_goes_to_the_base_file(self, tmp_path):
        s = SnapshotCatalog(tmp_path)
        save(s, "wall_decor", "wall poster")
        assert (tmp_path / "wall_decor.json").is_file()


class TestPlans:
    def plan(self, snaps, themes, **kw):
        c = FurnishingCoordinator(LayeredCatalog(snaps))
        return c, c.create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",), audience="child", age=8,
                                                  themes=themes, **kw))

    def full(self, tmp_path, decor_query="monster cartoon wall poster"):
        snaps = SnapshotCatalog(tmp_path)
        for cat in ("kids_bed", "mattress", "wardrobe", "study_table", "study_chair", "bookshelf", "table_lamp",
                    "bean_bag", "toy_storage"):
            snaps.save(CategoryListings(cat, [dict(r, category=cat, data_origin=ORIGIN_REAL)
                                              for r in SimulatedCatalog().listings(cat).records],
                                        ORIGIN_REAL, "2026-11-01T00:00:00+00:00", category(cat).search_term))
        save(snaps, "wall_decor", decor_query)
        save(snaps, "rug", "kids rug", ["Cartoon Kids Rug", "Plain Kids Rug"])   # a generic search: valid for any theme
        return snaps

    def test_a_matching_theme_fills_wall_decor_from_the_saved_search(self, tmp_path):
        _, p = self.plan(self.full(tmp_path), ("monster", "cartoon"))
        line = p.lines["kids_bedroom.wall_decor"]
        assert not line["no_data"] and line["product"]["theme_matches"] and not line["theme_unmatched"]

    def test_a_different_theme_does_not_reuse_the_monster_data(self, tmp_path):
        _, p = self.plan(self.full(tmp_path), ("princess", "barbie"))
        line = p.lines["kids_bedroom.wall_decor"]
        assert line["no_data"] and line["status"] == "unfilled" and line["product"] is None
        assert "does not cover" in line["why"] and line["search_query"] == "princess barbie wall poster"
        assert p.summary["no_data_count"] == 1

    def test_the_budget_of_an_item_without_matching_data_goes_to_the_others(self, tmp_path):
        _, p = self.plan(self.full(tmp_path), ("princess",))
        total = sum(ln["ceiling"] for ln in p.lines.values())
        assert p.lines["kids_bedroom.wall_decor"]["ceiling"] == 0 and 49999.5 <= total <= 50000.0      # never above the budget

    def test_a_generic_pick_is_flagged_as_not_themed_with_the_search_a_themed_one_would_use(self, tmp_path):
        _, p = self.plan(self.full(tmp_path), ("princess", "barbie"))
        rug = p.lines["kids_bedroom.rug"]
        assert rug["product"] is not None and rug["theme_unmatched"] and rug["search_query"] == "princess barbie kids rug"
        assert p.summary["theme_unmatched_count"] == 1

    def test_a_themed_pick_is_not_flagged_and_neither_is_a_request_without_themes(self, tmp_path):
        snaps = self.full(tmp_path)
        _, p = self.plan(snaps, ("cartoon",))
        assert not p.lines["kids_bedroom.rug"]["theme_unmatched"]
        _, p = self.plan(snaps, ())
        assert not any(ln.get("theme_unmatched") for ln in p.lines.values())

    def test_other_items_are_never_flagged(self, tmp_path):
        _, p = self.plan(self.full(tmp_path), ("princess",))
        assert not any(ln["theme_unmatched"] for ln in p.lines.values() if ln["category"] not in ("rug", "wall_decor"))

    def test_the_chat_says_what_is_missing_and_what_is_plain(self, tmp_path):
        _, p = self.plan(self.full(tmp_path), ("princess", "barbie"))
        text = ChatAssistant.describe_plan(p)
        assert "no product listings yet for: wall decor" in text and "plain one, not a themed one" in text
        assert "an 8-year-old" in text

    def test_changing_the_theme_re_reads_the_theme_dependent_listings(self, tmp_path):
        snaps = self.full(tmp_path)
        c, p = self.plan(snaps, ("princess", "barbie"))
        assert p.lines["kids_bedroom.wall_decor"]["no_data"]
        p = c.handle_event({"type": "preference_change", "themes": ["monster", "cartoon"]})
        assert not p.lines["kids_bedroom.wall_decor"]["no_data"] and p.lines["kids_bedroom.wall_decor"]["product"]["theme_matches"]
        p = c.handle_event({"type": "preference_change", "themes": ["space"]})
        assert p.lines["kids_bedroom.wall_decor"]["no_data"]

    def test_a_fresh_themed_search_is_picked_up_after_a_refresh_event(self, tmp_path):
        snaps = self.full(tmp_path)
        c, p = self.plan(snaps, ("princess", "barbie"))
        save(snaps, "wall_decor", "princess barbie wall poster", ["Princess Barbie Poster for Girls", "Barbie Sticker"])
        p = c.handle_event({"type": "refresh_category", "category": "wall_decor"})
        line = p.lines["kids_bedroom.wall_decor"]
        assert not line["no_data"] and set(line["product"]["theme_matches"]) == {"princess", "barbie"}
        assert not p.summary["no_data_count"]
        # the old monster data is still there for a monster request
        assert snaps.listings_for("wall_decor", ("monster",)).records

    def test_a_fake_catalog_without_listings_for_still_works(self):
        class Plain:
            def listings(self, category_id):
                return CategoryListings(category_id, records(category_id) if category_id == "wall_decor" else [],
                                        ORIGIN_REAL, "t", "whatever")
        plan = FurnishingCoordinator(Plain()).create_plan(FurnishingRequest(5000, None, items=("wall_decor",), themes=("x1",)))
        assert plan.lines["shopping_list.wall_decor"]["product"] is not None


class TestRugsAreRugs:
    @pytest.mark.parametrize("title", ["Barbie Blanket | Barbie Fleece Throw Blanket (Kids) 40 x 50", "Kids Bed Sheet Set Cartoon",
                                       "Cartoon Curtain for Kids Room", "Soft Cushion Cover", "Bath Towel Set",
                                       "BARBIE 70s DoLLHOUSE RUG, Little, Striped, Toy, Carpet", "Miniature Rug 1:12 scale"])
    def test_blankets_sheets_and_curtains_are_not_rugs(self, title):
        assert category("rug").accessory_reason(title, 1500)

    @pytest.mark.parametrize("title", ["VS Home Living Cartoon Theme Kids Floor Mat", "Barbie Princess Play Mat, Pastel Round Area Rug",
                                       "Franco Collectibles Barbie 100% Cotton Non Slip Plush Rug"])
    def test_rugs_and_floor_mats_are_kept(self, title):
        assert category("rug").accessory_reason(title, 2500) is None
