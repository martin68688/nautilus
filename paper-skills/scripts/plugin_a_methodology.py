"""Plugin A: Download PDFs, extract methodology via LLM, save to methodology_kb/"""
import os, re, time, json, argparse
import urllib.request
import pymupdf4llm
from openai import OpenAI
from pathlib import Path

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

PROMPT = """Extract techniques from this ML/NLP paper. For each technique or design choice, identify whether it had a positive, negative, or neutral effect on results.

Return JSON with a "techniques" array. Each item:
- name: short technique name
- description: what it is
- effect: "positive" | "negative" | "neutral"
- delta: quantitative change if mentioned (e.g. "+2.3 F1"), or descriptive ("outperforms baseline")
- evidence: direct quote from paper supporting the claim
- condition: when/where this applies

Paper text:
{text}

Return only valid JSON: {{"techniques": [...]}}"""


def get_pdf_url(source_url: str) -> str:
    # https://aclanthology.org/2024.naacl-long.113/ -> .pdf
    return source_url.rstrip("/") + ".pdf"


def download_pdf(url: str, dest: Path, retries: int = 3) -> bool:
    for i in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            if i < retries - 1:
                time.sleep(2 ** i)
            else:
                print(f"  Download failed {url}: {e}")
    return False


def extract_text(pdf_path: Path) -> str:
    text = pymupdf4llm.to_markdown(str(pdf_path))
    return text[:64000]


def render_methodology(title: str, source: str, techniques: list) -> str:
    lines = [f"# {title}\n", f"**Source**: {source}\n"]
    for t in techniques:
        if not t.get("name"):
            continue
        effect = t.get("effect", "neutral").upper()
        lines.append(f"## [{effect}] {t['name']}")
        lines.append(f"{t.get('description', '')}\n")
        lines.append(f"**Delta**: {t.get('delta', 'N/A')}")
        lines.append(f"**Condition**: {t.get('condition', 'N/A')}\n")
        lines.append(f"**Evidence**: \"{t.get('evidence', '')}\"\n")
    return "\n".join(lines)


def extract_methodology(text: str) -> list:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
        temperature=0,
    )
    raw = resp.choices[0].message.content
    raw = re.sub(r"^```json\s*|```$", "", raw.strip(), flags=re.MULTILINE)
    return json.loads(raw).get("techniques", [])


def process_category(category_dir: Path, output_dir: Path):
    refs_dir = category_dir / "references"
    if not refs_dir.exists():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path("cache/plugin_a_pdfs")
    tmp.mkdir(parents=True, exist_ok=True)

    ref_files = sorted(refs_dir.glob("*.md"))
    total = len(ref_files)
    done = skipped = failed = 0

    for i, ref_file in enumerate(ref_files, 1):
        out_file = output_dir / ref_file.name.replace(".md", "_methodology.md")
        prefix = f"[{i}/{total}]"
        if out_file.exists():
            skipped += 1
            print(f"{prefix} Skip: {ref_file.name}")
            continue

        content = ref_file.read_text()
        m = re.search(r'source:\s*"([^"]+)"', content)
        if not m:
            skipped += 1
            continue
        source_url = m.group(1)

        if "aclanthology.org" not in source_url:
            skipped += 1
            print(f"{prefix} Skip (non-ACL): {ref_file.name}")
            continue

        pdf_url = get_pdf_url(source_url)
        pdf_path = tmp / (ref_file.stem + ".pdf")

        print(f"{prefix} Downloading {ref_file.name}...", end=" ", flush=True)
        if not download_pdf(pdf_url, pdf_path):
            failed += 1
            print("FAILED")
            continue
        print("OK", end=" ", flush=True)

        text = extract_text(pdf_path)
        print("extracted", end=" ", flush=True)

        try:
            techniques = extract_methodology(text)
        except Exception as e:
            failed += 1
            print(f"LLM FAILED: {e}")
            continue

        title_m = re.search(r'title:\s*"([^"]+)"', content)
        title = title_m.group(1) if title_m else ref_file.stem
        out_file.write_text(render_methodology(title, source_url, techniques))
        done += 1
        print(f"saved ({len(techniques)} techniques)")
        time.sleep(0.5)

    print(f"\nDone: {done} saved, {skipped} skipped, {failed} failed / {total} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--category", required=True, help="category folder name")
    args = parser.parse_args()

    base = Path("output") / f"{args.venue}-{args.year}"
    category_dir = base / args.category
    output_dir = Path("methodology_kb") / f"{args.venue}-{args.year}" / args.category

    process_category(category_dir, output_dir)
    print("Done.")
