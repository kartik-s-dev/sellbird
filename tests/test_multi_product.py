from harvey.agents.scout import Scout
"""Tests for multi-product data models."""

import pytest
from pydantic import ValidationError

from harvey.models.product import Product
from harvey.models.company import Company
from harvey.models.prospect import Prospect
from harvey.state import StateManager
from harvey.trainer import Trainer


def test_product_creation():
    product = Product(
        id="product-001",
        name="CRM Software",
        description="CRM for growing businesses",
        pricing="$99/month",
        key_benefits=["Automation", "Reporting"],
        objection_responses={"expensive": "It saves operational costs"},
    )

    assert product.id == "product-001"
    assert product.name == "CRM Software"
    assert product.pricing == "$99/month"
    assert len(product.key_benefits) == 2


def test_product_defaults():
    product = Product(
        id="product-002",
        name="Analytics",
        description="Business analytics platform",
        pricing="$49/month",
    )

    assert product.key_benefits == []
    assert product.objection_responses == {}
    assert product.is_active is True


def test_product_rejects_empty_name():
    with pytest.raises(ValidationError):
        Product(
            id="product-003",
            name="",
            description="Description",
            pricing="$10",
        )


def test_products_have_independent_data():
    first = Product(
        id="p1",
        name="CRM",
        description="CRM platform",
        pricing="$99",
    )
    second = Product(
        id="p2",
        name="Analytics",
        description="Analytics platform",
        pricing="$49",
    )

    assert first.id != second.id
    assert first.name != second.name
    assert first.key_benefits is not second.key_benefits



@pytest.mark.asyncio
async def test_product_database_crud(tmp_path):
    state = StateManager(str(tmp_path / "products.db"))
    await state.init_db()

    product = Product(
        id="product-db-001",
        name="CRM Software",
        description="CRM for growing businesses",
        pricing="$99/month",
        key_benefits=["Automation", "Reporting"],
        objection_responses={"expensive": "It saves operational costs"},
        offer={"goal": "book_call"},
        icp={"industries": ["Technology"], "company_size": "50-200"},
        business_context="We sell CRM software to growing businesses.",
        qualification_rules={"minimum_employees": 50},
        sales_context="Focus on operational efficiency.",
    )

    product_id = await state.add_product(product)
    assert product_id == "product-db-001"

    saved = await state.get_product(product_id)
    assert saved is not None
    assert saved.name == "CRM Software"
    assert saved.key_benefits == ["Automation", "Reporting"]
    assert saved.icp["industries"] == ["Technology"]
    assert saved.qualification_rules["minimum_employees"] == 50

    updated = await state.update_product(
        product_id,
        name="Updated CRM",
        pricing="$149/month",
    )
    assert updated is True

    saved_again = await state.get_product(product_id)
    assert saved_again.name == "Updated CRM"
    assert saved_again.pricing == "$149/month"

    products = await state.list_products()
    assert len(products) == 1
    assert products[0].id == product_id


@pytest.mark.asyncio
async def test_product_active_filter(tmp_path):
    state = StateManager(str(tmp_path / "active_products.db"))
    await state.init_db()

    await state.add_product(
        Product(id="active-001", name="Active Product")
    )
    await state.add_product(
        Product(id="inactive-001", name="Inactive Product", is_active=False)
    )

    active_products = await state.list_products(active_only=True)

    assert len(active_products) == 1
    assert active_products[0].id == "active-001"


@pytest.mark.asyncio
async def test_update_missing_product_returns_false(tmp_path):
    state = StateManager(str(tmp_path / "missing_product.db"))
    await state.init_db()

    result = await state.update_product(
        "does-not-exist",
        name="Unknown Product",
    )

    assert result is False
@pytest.mark.asyncio
async def test_link_product_to_company(tmp_path):
    state = StateManager(str(tmp_path / "product_company.db"))
    await state.init_db()

    await state.add_product(Product(id="p1", name="CRM"))
    await state.add_company(Company(id="c1", name="Acme"))

    result = await state.link_product_company("p1", "c1")
    assert result is True

    # Repeating the same link must not create a duplicate.
    duplicate = await state.link_product_company("p1", "c1")
    assert duplicate is False


@pytest.mark.asyncio
async def test_link_product_to_missing_company_or_product(tmp_path):
    state = StateManager(str(tmp_path / "missing_company.db"))
    await state.init_db()

    await state.add_product(Product(id="p1", name="CRM"))

    assert await state.link_product_company("p1", "missing") is False
    assert await state.link_product_company("missing", "c1") is False


@pytest.mark.asyncio
async def test_link_product_to_prospect(tmp_path):
    state = StateManager(str(tmp_path / "product_prospect.db"))
    await state.init_db()

    await state.add_product(Product(id="p1", name="CRM"))
    await state.add_company(Company(id="c1", name="Acme"))
    await state.add_prospect(
        Prospect(
            id="prospect1",
            company_id="c1",
            first_name="Jane",
            last_name="Doe",
            title="CEO",
        )
    )

    result = await state.link_product_prospect("p1", "prospect1")
    assert result is True

    duplicate = await state.link_product_prospect("p1", "prospect1")
    assert duplicate is False


@pytest.mark.asyncio
async def test_link_product_to_missing_prospect_or_product(tmp_path):
    state = StateManager(str(tmp_path / "missing_prospect.db"))
    await state.init_db()

    await state.add_product(Product(id="p1", name="CRM"))

    assert await state.link_product_prospect("p1", "missing") is False
    assert await state.link_product_prospect("missing", "prospect1") is False



@pytest.mark.asyncio
async def test_trainer_saves_product_to_isolated_database(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.state = StateManager(str(tmp_path / "trainer_test.db"))

    product_info = {
        "product_name": "Test CRM",
        "product_description": "A CRM for small businesses",
        "pricing": "$20/month",
        "key_benefits": ["Simple setup", "Contact management"],
        "value_proposition": "Manage customer relationships",
        "features": ["Contact tracking"],
        "differentiators": ["Easy to use"],
        "use_cases": ["Sales tracking"],
        "company_name": "Example Inc",
        "target_audience": "Small businesses",
        "tone": "Professional",
        "social_proof": [],
        "integrations": [],
    }

    icp_info = {}
    objections = {"Too expensive": "Explain the return on investment"}

    await trainer.state.init_db()
    product_id = await trainer._save_product(
        "example.com", product_info, icp_info, objections
    )

    saved = await trainer.state.get_product(product_id)

    assert saved is not None
    assert saved.name == "Test CRM"
    assert saved.description == "A CRM for small businesses"
    assert saved.key_benefits == ["Simple setup", "Contact management"]
    assert saved.icp == {}
    assert saved.objection_responses == objections
    assert product_id.startswith("prod_")


@pytest.mark.asyncio
async def test_trainer_rejects_empty_product_name(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.state = StateManager(str(tmp_path / "trainer_empty_test.db"))

    await trainer.state.init_db()

    with pytest.raises(ValueError, match="product name is empty"):
        await trainer._save_product(
            "example.com", {"product_name": "  "}, {}, {}
        )



@pytest.mark.asyncio
async def test_products_retrieve_only_their_own_companies_and_prospects(tmp_path):
    state = StateManager(str(tmp_path / "product_isolation.db"))
    await state.init_db()

    await state.add_product(Product(id="product-a", name="CRM"))
    await state.add_product(Product(id="product-b", name="Analytics"))

    await state.add_company(Company(id="company-a", name="Company A"))
    await state.add_company(Company(id="company-b", name="Company B"))

    await state.add_prospect(
        Prospect(
            id="prospect-a",
            company_id="company-a",
            first_name="Alice",
            last_name="Example",
            title="Sales Manager",
        )
    )
    await state.add_prospect(
        Prospect(
            id="prospect-b",
            company_id="company-b",
            first_name="Bob",
            last_name="Example",
            title="Data Analyst",
        )
    )

    assert await state.link_product_company("product-a", "company-a")
    assert await state.link_product_company("product-b", "company-b")
    assert await state.link_product_prospect("product-a", "prospect-a")
    assert await state.link_product_prospect("product-b", "prospect-b")

    companies_a = await state.get_product_companies("product-a")
    companies_b = await state.get_product_companies("product-b")
    prospects_a = await state.get_product_prospects("product-a")
    prospects_b = await state.get_product_prospects("product-b")

    assert [company.id for company in companies_a] == ["company-a"]
    assert [company.id for company in companies_b] == ["company-b"]
    assert [prospect.id for prospect in prospects_a] == ["prospect-a"]
    assert [prospect.id for prospect in prospects_b] == ["prospect-b"]

    assert await state.get_product_companies("unknown-product") == []
    assert await state.get_product_prospects("unknown-product") == []
from harvey.agents.scout import Scout


@pytest.mark.asyncio
async def test_scout_links_created_prospect_to_its_product(tmp_path):
    state = StateManager(str(tmp_path / "scout_product.db"))
    await state.init_db()

    await state.add_product(Product(id="p1", name="CRM"))
    await state.add_company(Company(id="c1", name="Acme"))

    await state.add_prospect(
        Prospect(
            id="scout-prospect-1",
            company_id="c1",
            first_name="Jane",
            last_name="Doe",
            title="CEO",
        )
    )

    scout = Scout.__new__(Scout)
    scout.state = state
    scout.product_id = "p1"

    await scout._link_prospect_to_product("scout-prospect-1")

    linked = await state.get_product_prospects("p1")

    assert [prospect.id for prospect in linked] == ["scout-prospect-1"]


@pytest.mark.asyncio
async def test_scout_without_product_id_does_not_create_product_link(tmp_path):
    state = StateManager(str(tmp_path / "scout_legacy.db"))
    await state.init_db()

    await state.add_product(Product(id="p1", name="CRM"))
    await state.add_company(Company(id="c1", name="Acme"))

    await state.add_prospect(
        Prospect(
            id="legacy-prospect-1",
            company_id="c1",
            first_name="John",
            last_name="Doe",
            title="Manager",
        )
    )

    scout = Scout.__new__(Scout)
    scout.state = state
    scout.product_id = None

    await scout._link_prospect_to_product("legacy-prospect-1")

    linked = await state.get_product_prospects("p1")

    assert linked == []