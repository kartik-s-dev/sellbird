
"""Writer — crafts personalized email sequences.

With a native mail provider (Gmail/SMTP), email 1 is written PER PROSPECT
from grounded facts — the company's own description, detected tech stack,
and buying signals — instead of one merge-tag template shared by a whole
batch.

Steps 2-3 stay campaign-level. Every draft still passes the deterministic
pre-send gate.

Product-aware mode:
- Processes only prospects explicitly linked to the selected product.
- Stores the product ID on every generated campaign.
- Uses the product-specific HarveyConfig supplied by the orchestrator.
- Preserves legacy global behaviour when no product_id is supplied.
"""

import json
import logging
from datetime import datetime, timezone

from harvey.brain import Brain
from harvey.config import HarveyConfig
from harvey.integrations.mail_provider import NATIVE_PROVIDERS
from harvey.models.campaign import Campaign, EmailStep
from harvey.state import StateManager

logger = logging.getLogger("harvey.writer")

# Native mode: one LLM call per prospect for email 1, so batches stay small.
NATIVE_BATCH_CAP = 8
LEGACY_BATCH_CAP = 50


class Writer:
    def __init__(
        self,
        brain: Brain,
        state: StateManager,
        config: HarveyConfig,
        env=None,
        product_id: str | None = None,
    ):
        self.brain = brain
        self.state = state
        self.config = config
        self.env = env
        self.product_id = str(product_id).strip() if product_id else None
        self.skills = ""

    @property
    def is_native(self) -> bool:
        return self.config.channels.email.provider in NATIVE_PROVIDERS

    def _session_id(self, task: str) -> str:
        """Keep product-specific LLM sessions separate."""
        if self.product_id:
            return f"harvey-writer-{self.product_id}-{task}"
        return f"harvey-writer-{task}"

    async def _get_new_prospects(self) -> list:
        """Fetch only eligible prospects for this Writer instance."""
        if self.product_id:
            linked_prospects = await self.state.get_product_prospects(
                self.product_id
            )
            prospects = [
                prospect
                for prospect in linked_prospects
                if prospect.status == "new"
            ]
            logger.info(
                "Writer: Product mode enabled for %s; found %d new linked prospect(s).",
                self.product_id,
                len(prospects),
            )
        else:
            # Preserve existing global behaviour.
            prospects = await self.state.get_prospects_by_status("new")

        # Protect against accidental duplicate associations.
        unique = {}
        for prospect in prospects:
            if prospect.id:
                unique[prospect.id] = prospect

        return list(unique.values())

    async def run(self):
        """Create email campaigns for prospects that need outreach."""
        logger.info("Writer: Crafting email campaigns...")

        self.skills = self.brain.load_skills_for_agent("writer") or ""

        try:
            new_prospects = await self._get_new_prospects()
        except Exception:
            logger.exception("Writer: Failed to retrieve prospects.")
            return

        if not new_prospects:
            logger.info("Writer: No new prospects to write for.")
            return

        # Do not spend LLM calls or sending reputation on guessed/invalid
        # email addresses.
        deliverable = {"verified", "risky"}
        prospects_with_email = [
            prospect
            for prospect in new_prospects
            if prospect.email
            and (prospect.email_status or "guess") in deliverable
        ]

        if not prospects_with_email:
            logger.info(
                "Writer: No prospects with deliverable (verified/risky) emails yet."
            )
            return

        batches = self._group_prospects(prospects_with_email)

        for batch_name, prospects in batches.items():
            if not prospects:
                continue

            logger.info(
                "Writer: Creating campaign '%s' for %d prospects.",
                batch_name,
                len(prospects),
            )

            try:
                sequence = await self._write_sequence(prospects)
            except Exception:
                logger.exception(
                    "Writer: Sequence generation failed for %s.", batch_name
                )
                continue

            if not sequence:
                logger.warning(
                    "Writer: Failed to generate sequence for %s.", batch_name
                )
                continue

            campaign = Campaign(
                id="",
                name=batch_name,
                channel="email",
                sequence=sequence,
                prospect_ids=[prospect.id for prospect in prospects],
                status="draft",
                product_id=self.product_id or "",
            )

            try:
                campaign_id = await self.state.add_campaign(campaign)
            except Exception:
                logger.exception(
                    "Writer: Failed to save campaign %s.", batch_name
                )
                continue

            if not campaign_id:
                logger.error(
                    "Writer: Campaign %s was not saved; prospects remain unchanged.",
                    batch_name,
                )
                continue

            # Mark prospects only after campaign creation succeeds.
            for prospect in prospects:
                try:
                    await self.state.update_prospect_status(
                        prospect.id, "queued"
                    )
                except Exception:
                    logger.exception(
                        "Writer: Could not queue prospect %s.",
                        prospect.id,
                    )

            # Native providers get individually grounded first-touch drafts.
            if self.is_native:
                try:
                    await self._personalize_first_emails(
                        campaign_id, prospects
                    )
                except Exception:
                    logger.exception(
                        "Writer: Personalization failed for campaign %s.",
                        campaign_id,
                    )

            try:
                await self.state.log_action(
                    action_type="write_campaign",
                    agent="writer",
                    details={
                        "campaign_id": campaign_id,
                        "campaign_name": batch_name,
                        "product_id": self.product_id or "",
                        "prospect_count": len(prospects),
                        "steps": len(sequence),
                    },
                )
            except Exception:
                logger.exception(
                    "Writer: Failed to log campaign creation %s.",
                    campaign_id,
                )

            logger.info(
                "Writer: Campaign '%s' created with %d emails for %d prospects.",
                batch_name,
                len(sequence),
                len(prospects),
            )

    async def _personalize_first_emails(
        self, campaign_id: str, prospects: list
    ):
        """Draft grounded, per-prospect email 1 and stage it in the outbox."""
        require_approval = getattr(
            self.config.channels.email, "require_approval", True
        )
        status = "pending_review" if require_approval else "approved"
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        provider = self.config.channels.email.provider

        drafted = 0

        for prospect in prospects:
            try:
                draft = await self._write_personal_email(prospect)
            except Exception:
                logger.exception(
                    "Writer: Personal draft failed for prospect %s.",
                    prospect.id,
                )
                continue

            if not draft:
                continue

            try:
                item_id = await self.state.add_outbox_item(
                    prospect_id=prospect.id,
                    campaign_id=campaign_id,
                    step=1,
                    to_email=prospect.email,
                    subject=draft["subject"],
                    body=draft["body"],
                    send_at=now,
                    status=status,
                    provider=provider,
                )
            except Exception:
                logger.exception(
                    "Writer: Could not stage personal draft for prospect %s.",
                    prospect.id,
                )
                continue

            if item_id:
                drafted += 1

        if drafted:
            logger.info(
                "Writer: Drafted %d personalized first email(s) (%s).",
                drafted,
                "awaiting approval" if require_approval else "approved",
            )

    async def _write_personal_email(self, prospect) -> dict | None:
        """Generate one grounded first-touch email for one prospect."""
        facts = [
            f"- Name: {prospect.full_name()}",
            f"- Title: {prospect.title}",
            f"- Company: {prospect.company}",
        ]

        if prospect.industry:
            facts.append(f"- Industry: {prospect.industry}")

        company = None
        if prospect.company_id:
            try:
                company = await self.state.get_company(prospect.company_id)
            except Exception:
                logger.warning(
                    "Writer: Could not load company facts for prospect %s.",
                    prospect.id,
                )

        if company:
            if company.description:
                facts.append(
                    f"- What the company says about itself: {company.description}"
                )

            if company.tech_stack:
                tools = ", ".join(company.tech_stack[:6])
                facts.append(f"- Tools detected on their website: {tools}")

            for signal in (company.signals or [])[:3]:
                if not isinstance(signal, dict):
                    continue
                facts.append(
                    f"- Signal ({signal.get('type', 'signal')}): "
                    f"{signal.get('detail', '')}"
                )

        if prospect.personalization_notes:
            facts.append(
                f"- Research notes: {prospect.personalization_notes}"
            )

        product = self.config.product
        persona = self.config.persona
        benefits = product.key_benefits or []

        prompt = self.brain.load_prompt(
            "writer",
            product_name=product.name,
            product_description=product.description,
            product_benefits="\n".join(
                f"- {benefit}" for benefit in benefits
            ),
            product_pricing=product.pricing,
            persona_name=persona.name,
            persona_company=persona.company,
            persona_role=persona.role,
            persona_tone=persona.tone,
        ) or ""

        if self.skills:
            prompt += "\n\n" + self.skills

        prompt += f"""

Write ONE cold email (the very first touch) to this specific person.

FACTS — everything known about this person and company. Every claim about
the prospect in your email must come from these facts. If a fact isn't
listed here, you don't know it. Do not invent funding rounds, mutual
contacts, metrics, achievements, or any other details.

{chr(10).join(facts)}

Requirements:
- Under 75 words.
- One specific observation supported by the FACTS above.
- Ask one relevant question.
- No pitch and no product name.
- Use the person's actual name and company, not merge variables.
- Subject: lowercase, 2-5 words.
- Follow every STRICT EMAIL RULE above.

Return ONLY JSON: {{"subject": "...", "body": "..."}}"""

        result = await self.brain.think_json(
            prompt,
            session_id=self._session_id("personal"),
            agent="writer",
            task="personal_email",
        )

        if not isinstance(result, dict):
            return None

        subject = result.get("subject")
        body = result.get("body")

        if not isinstance(subject, str) or not isinstance(body, str):
            return None

        subject = subject.strip()
        body = body.strip()

        if not subject or not body:
            return None

        return {
            "subject": subject[:120],
            "body": body[:2000],
        }

    async def _write_sequence(self, prospects: list) -> list[EmailStep]:
        """Generate a three-email campaign sequence."""
        prospect_summary = "\n".join(
            f"- {prospect.full_name()}, {prospect.title} at {prospect.company}"
            + (
                f" | Notes: {prospect.personalization_notes}"
                if prospect.personalization_notes
                else ""
            )
            for prospect in prospects[:5]
        )

        product = self.config.product
        persona = self.config.persona
        benefits = product.key_benefits or []

        prompt = self.brain.load_prompt(
            "writer",
            product_name=product.name,
            product_description=product.description,
            product_benefits="\n".join(
                f"- {benefit}" for benefit in benefits
            ),
            product_pricing=product.pricing,
            persona_name=persona.name,
            persona_company=persona.company,
            persona_role=persona.role,
            persona_tone=persona.tone,
        )

        if not prompt:
            prompt = f"""You are {persona.name}, {persona.role} at {persona.company}.
Your tone is: {persona.tone}

Product: {product.name}
Description: {product.description}
Key benefits: {', '.join(benefits)}
Pricing: {product.pricing}"""

        if self.skills:
            prompt += "\n\n" + self.skills

        prompt += f"""

Write a 3-email cold outreach sequence for prospects like these:
{prospect_summary}

Requirements:
- Email 1: Personalized cold observation + one question. Under 75 words. No pitch.
- Email 2: Follow-up 3 days later. Different angle, one concrete proof point.
  Under 75 words. Do not invent proof points; use only supplied product facts.
- Email 3: Break-up email 4 days after that. Under 40 words. Gracious, not guilt-tripping.
- Use {{{{first_name}}}}, {{{{company}}}}, {{{{title}}}} as merge variables.
  Every email must use at least one.
- Email 1 must reference a relevant industry or role detail from the prospect context.
- Never invent prospect facts, customer results, metrics, testimonials, or proof.
- Never be pushy or salesy. Be consultative and value-driven.
- Subject lines: lowercase, 2-5 words.
- Follow every rule in the STRICT EMAIL RULES above. No exceptions.

Return ONLY a JSON array (no markdown fences, no commentary):
[
  {{"step": 1, "subject": "...", "body": "...", "delay_days": 0}},
  {{"step": 2, "subject": "...", "body": "...", "delay_days": 3}},
  {{"step": 3, "subject": "...", "body": "...", "delay_days": 4}}
]"""

        result = await self.brain.think_json(
            prompt,
            session_id=self._session_id("sequence"),
            agent="writer",
            task="write_sequence",
        )

        return self._parse_sequence(result)

    def _parse_sequence(self, result) -> list[EmailStep]:
        """Coerce LLM output into a validated EmailStep list.

        Never raises. Accepts a direct list or a common wrapper dictionary.
        Invalid steps are dropped.
        """
        if isinstance(result, dict):
            for key in ("emails", "sequence", "steps", "campaign"):
                if isinstance(result.get(key), list):
                    result = result[key]
                    break

        if not isinstance(result, list) or not result:
            logger.error(
                "Writer: Brain did not return an email array (got %s).",
                type(result).__name__,
            )
            return []

        steps: list[EmailStep] = []

        for index, raw in enumerate(result[:5]):
            if not isinstance(raw, dict):
                logger.warning(
                    "Writer: Skipping non-dict step at index %d.", index
                )
                continue

            subject = raw.get("subject")
            body = raw.get("body")

            if not isinstance(subject, str) or not isinstance(body, str):
                logger.warning(
                    "Writer: Skipping step %d with invalid subject/body types.",
                    index + 1,
                )
                continue

            subject = subject.strip()
            body = body.strip()

            if not subject or not body:
                logger.warning(
                    "Writer: Skipping step %d with empty subject/body.",
                    index + 1,
                )
                continue

            try:
                delay = max(
                    0, int(raw.get("delay_days", 3 if steps else 0))
                )
            except (TypeError, ValueError, OverflowError):
                delay = 3 if steps else 0

            try:
                steps.append(
                    EmailStep(
                        step=len(steps) + 1,
                        subject=subject[:200],
                        body=body[:10000],
                        delay_days=delay,
                    )
                )
            except Exception:
                logger.exception(
                    "Writer: Skipping invalid step %d.", index + 1
                )

        if steps and steps[0].delay_days != 0:
            steps[0].delay_days = 0

        if len(steps) != 3:
            logger.warning(
                "Writer: Expected 3 emails, got %d.", len(steps)
            )

        return steps

    def _group_prospects(self, prospects: list) -> dict[str, list]:
        """Group prospects into bounded campaign batches by industry."""
        batches: dict[str, list] = {}

        for prospect in prospects:
            industry = prospect.industry or "general"
            key = f"{industry}-outreach"
            batches.setdefault(key, []).append(prospect)

        cap = NATIVE_BATCH_CAP if self.is_native else LEGACY_BATCH_CAP

        # Only the first batch-sized group is returned. Remaining prospects
        # retain their "new" status and can be processed in a later cycle.
        return {
            key: group[:cap]
            for key, group in batches.items()
        }
