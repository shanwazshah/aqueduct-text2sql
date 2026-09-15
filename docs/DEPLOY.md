# Deploying the demo

The app runs on [Streamlit Community Cloud](https://share.streamlit.io) for free.
It needs a model, and Cloud has no GPU — so the hosted demo talks to an
OpenAI-compatible API instead of local Ollama. Nothing in the code changes; the
backend is a `base_url`.

**Three steps, about fifteen minutes.**

---

## 1 · Get a model endpoint

[Groq](https://console.groq.com) is the recommended one: free tier, fast, and
OpenAI-compatible so it needs no code change.

1. Sign up at **console.groq.com**
2. **API Keys** → **Create API Key** → copy it (shown once)
3. **Models** (in the console) → note an available model id

Any OpenAI-compatible provider works — OpenAI, Together, Fireworks, or your own
vLLM. Only the `base_url`, key and model name differ.

---

## 2 · Deploy

1. Go to **share.streamlit.io** and sign in with GitHub
2. **Create app** → **Deploy a public app from GitHub**
3. Fill in:
   - Repository: `shanwazshah/aqueduct-text2sql`
   - Branch: `master`
   - Main file path: **`ui/app.py`**
4. Open **Advanced settings** → **Secrets**, and paste:

```toml
AQ_BASE_URL      = "https://api.groq.com/openai/v1"
AQ_API_KEY       = "gsk_your_key_here"
AQ_MODEL_SQL     = "llama-3.3-70b-versatile"
AQ_MODEL_CRITIC  = "llama-3.3-70b-versatile"
AQ_MODEL_LEAD    = "llama-3.3-70b-versatile"
AQ_MODEL_ANALYST = "llama-3.3-70b-versatile"
```

5. **Deploy**

> **The model id is the thing that breaks this.** Providers rename and retire
> models, so a name copied from a blog post is often already dead. Take it from
> the provider's own model list, not from here — the first deploy of this app
> failed with `The model llama-3.3-70b-versatile does not exist or you do not
> have access to it`. The app now says exactly that when it happens, and which
> secret to change.

First build takes a few minutes. The app seeds its own demo database on first
run, so there is nothing to upload.

---

## 3 · Check it

Open the URL and ask *"How many employees are in each department?"*

A blue banner should say it is running against a hosted model. If it says the
endpoint did not respond, the key or the model id is wrong — Streamlit's **Manage
app → logs** will say which.

---

## Notes worth knowing

**Secrets are not in the repository.** They live in Streamlit's settings.
`.gitignore` covers `.env` and `.streamlit/secrets.toml`, and the key should
never be committed. If one leaks, revoke it in the Groq console rather than
rewriting history.

**Which strategies will work.** `direct` needs only plain chat and works
anywhere. `chain`, `orchestrator` and the deep agent use JSON-schema structured
output, which not every provider implements — if one errors on a hosted backend,
that is why, and `direct` is both the fallback and the strategy that wins the
benchmark anyway.

**Free tiers rate-limit.** Shared quota, so a burst can return 429. The app
reports it rather than hanging.

**A hosted demo is not the benchmark.** It runs a different model against a
five-table toy database. The BIRD numbers in the README were measured with
`qwen2.5-coder` on a Kaggle T4 and are not reproduced by this app — it exists to
show the agents working, not to demonstrate accuracy.

---

## Running it locally instead

No key, no hosting, and it is the configuration every number in the README came
from:

```bash
ollama pull qwen2.5-coder:3b
streamlit run ui/app.py
```
