# FAQ

**Q: What if I do not have an `OPENAI_API_KEY`?**
A: Add it to `.env`; without it, the system falls back to local stubs for development.

**Q: Where do cost summaries live?**
A: Each `run_mode` response includes a `cost_summary` with token events.

**Q: How do I add a new domain?**
A: Create a pack in `configs/domains/`, update run docs, and extend validators if needed.
