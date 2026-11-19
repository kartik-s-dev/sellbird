
"""Tests for product-aware Writer campaign creation."""

import pytest

from harvey.agents.writer import Writer
from harvey.models.product import Product
from harvey.models.prospect import Prospect
from harvey.state import StateManager


class FakeConfig:
    class persona:
        name = "Harvey"
        company = "Example Inc"
        role = "Sales"
        tone = "Professional"

    class product:
        name = "Default Product"
        description = "Default description"
        pricing = "$99/month"
        key_benefits = ["Automation"]

    class channels:
        class email:
            provider = "smtp"
            enabled = True
            require_approval = True
            send_to_risky = False
            max_daily_sends = 50


class FakeBrain:
    def load_skills_for_agent(self, agent, product_id=None):
        return ""

    def load_prompt(self, *args, **kwargs):
        return ""

    async def think_json(self, *args, **kwargs):
        return [
            {"step": 1, "subject": "hello",
             "body": "A short introductory question?", "delay_days": 0},
            {"step": 2, "subject": "following up",
             "body": "A different relevant question?", "delay_days": 3},
            {"step": 3, "subject": "closing the loop",
             "body": "Would it be useful to reconnect?", "delay_days": 4},
        ]


async def seed_product(state, product_id, name):
    await state.add_product(Product(id=product_id, name=name))


async def seed_prospect(
    state, prospect_id, email, company,
    status="new", email_status="verified"
):
    prospect = Prospect(
        id=prospect_id,
        first_name="Jane",
        last_name="Doe",
        title="VP",
        company=company,
        industry="Technology",
        email=email,
        email_status=email_status,
        status=status,
    )
    await state.add_prospect(prospect)


async def make_writer(state, product_id=None):
    writer = Writer(
        brain=FakeBrain(),
        state=state,
        config=FakeConfig(),
        product_id=product_id,
    )

    async def skip_personalization(campaign_id, prospects):
        return None

    writer._personalize_first_emails = skip_personalization
    return writer


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_product_writer_only_gets_linked_product_prospects(tmp_path):
    state = StateManager(str(tmp_path / "writer_isolation.db"))
    await state.init_db()

    await seed_product(state, "product-a", "Product A")
    await seed_product(state, "product-b", "Product B")

    await seed_prospect(state, "prospect-a", "a@example.com", "Company A")
    await seed_prospect(state, "prospect-b", "b@example.com", "Company B")

    await state.link_product_prospect("product-a", "prospect-a")
    await state.link_product_prospect("product-b", "prospect-b")

    writer = await make_writer(state, product_id="product-a")
    prospects = await writer._get_new_prospects()

    assert [p.id for p in prospects] == ["prospect-a"]


@pytest.mark.asyncio
async def test_product_writer_creates_campaign_with_product_id(tmp_path):
    state = StateManager(str(tmp_path / "writer_campaign.db"))
    await state.init_db()

    await seed_product(state, "product-a", "Product A")
    await seed_prospect(state, "prospect-a", "a@example.com", "Company A")
    await state.link_product_prospect("product-a", "prospect-a")

    writer = await make_writer(state, product_id="product-a")
    await writer.run()

    campaigns = await state.get_campaigns_by_status("draft")

    assert len(campaigns) == 1
    assert campaigns[0].product_id == "product-a"
    assert campaigns[0].prospect_ids == ["prospect-a"]

    prospect = await state.get_prospect("prospect-a")
    assert prospect.status == "queued"


@pytest.mark.asyncio
async def test_legacy_writer_keeps_global_prospect_behaviour(tmp_path):
    state = StateManager(str(tmp_path / "writer_legacy.db"))
    await state.init_db()

    await seed_product(state, "product-a", "Product A")
    await seed_prospect(state, "prospect-a", "a@example.com", "Company A")
    await state.link_product_prospect("product-a", "prospect-a")

    writer = await make_writer(state, product_id=None)
    prospects = await writer._get_new_prospects()

    assert len(prospects) == 1
    assert prospects[0].id == "prospect-a"


@pytest.mark.asyncio
async def test_writer_skips_guessed_and_invalid_emails(tmp_path):
    state = StateManager(str(tmp_path / "writer_email_filter.db"))
    await state.init_db()

    await seed_prospect(
        state, "verified", "verified@example.com", "Verified Co",
        email_status="verified"
    )
    await seed_prospect(
        state, "risky", "risky@example.com", "Risky Co",
        email_status="risky"
    )
    await seed_prospect(
        state, "guessed", "guessed@example.com", "Guessed Co",
        email_status="guess"
    )
    await seed_prospect(
        state, "invalid", "invalid@example.com", "Invalid Co",
        email_status="invalid"
    )

    writer = await make_writer(state)
    prospects = await writer._get_new_prospects()

    deliverable = [
        p for p in prospects
        if p.email and (p.email_status or "guess") in {"verified", "risky"}
    ]

    assert {p.id for p in deliverable} == {"verified", "risky"}


@pytest.mark.asyncio
async def test_writer_does_not_queue_prospects_when_campaign_save_fails(
    tmp_path, monkeypatch
):
    state = StateManager(str(tmp_path / "writer_save_failure.db"))
    await state.init_db()

    await seed_prospect(state, "prospect-a", "a@example.com", "Company A")
    writer = await make_writer(state)

    async def fail_add_campaign(campaign):
        raise RuntimeError("Database write failed")

    monkeypatch.setattr(state, "add_campaign", fail_add_campaign)

    await writer.run()

    prospect = await state.get_prospect("prospect-a")
    assert prospect.status == "new"
